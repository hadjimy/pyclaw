#!/usr/bin/env python
# encoding: utf-8
r"""
Shu-Osher problem
====================

Solve the one-dimensional compressible Euler equations:

.. math::
    \rho_t + (\rho u)_x & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x & = 0 \\
    E_t + (u (E + p) )_x & = 0.

The initial condition corresponds to the Shu-Osher problem
in which a shock wave impacts a sinusoidally-varying density field.

This example also demonstrates:

 - how to use an arbitrary Runge-Kutta method by simply providing the
   Butcher coefficients of the method.
 - How to use a total fluctuation solver in SharpClaw
 - How to use characteristic decomposition with an evec() routine in SharpClaw
"""

import numpy as np
from clawpack import riemann
from clawpack.riemann.euler_with_efix_1D_constants import density, momentum, energy, num_eqn

gamma = 1.4  # Ratio of specific heats

# Coefficients of Runge-Kutta method
a = np.array([[0., 0., 0., 0., 0., 0., 0.],
              [.3772689153313680, 0., 0., 0., 0., 0., 0.],
              [.3772689153313680, .3772689153313680, 0., 0., 0., 0., 0.],
              [.2429952205373960, .2429952205373960, .2429952205373960, 0., 0., 0., 0.],
              [.1535890676951260, .1535890676951260, .1535890676951260, .2384589328462900, 0., 0., 0.]])
b = np.array([.206734020864804, .206734020864804, .117097251841844, .181802560120140, .287632146308408])
c = np.array([0., .3772689153313680, .7545378306627360, .7289856616121880, .6992261359316680])

def setup(use_petsc=False,iplot=False,htmlplot=False,outdir='./_output',solver_type='sharpclaw',
        kernel_language='Fortran',time_integrator='RK',use_char_decomp=True,tfluct_solver=False):

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    if kernel_language =='Python':
        rs = riemann.euler_1D_py.euler_roe_1D
    elif kernel_language =='Fortran':
        rs = riemann.euler_with_efix_1D

    if solver_type=='sharpclaw':
        solver = pyclaw.SharpClawSolver1D(rs)
        solver.time_integrator = time_integrator
        if solver.time_integrator == 'RK':
            solver.a, solver.b, solver.c = a, b, c
            solver.cfl_desired = 0.6
            solver.cfl_max = 0.7
        elif solver.time_integrator == 'DWRK':
            [solver.sspcoeff, solver.v, solver.a, solver.at, solver.c] = method_coeff
            solver.cfl_max = 0.5*solver.sspcoeff
        if use_char_decomp:
            try:
                import sharpclaw1               # Import custom Fortran code
                solver.fmod = sharpclaw1
                solver.tfluct_solver = tfluct_solver     # Use total fluctuation solver for efficiency
                if solver.tfluct_solver:
                    try:
                        import euler_tfluct
                        solver.tfluct = euler_tfluct
                    except ImportError:
                        import logging
                        logger = logging.getLogger()
                        logger.error('Unable to load tfluct solver, did you run make?')
                        print 'Unable to load tfluct solver, did you run make?'
                        raise
            except ImportError:
                import logging
                logger = logging.getLogger()
                logger.error('Unable to load sharpclaw1 solver, did you run make?')
                print 'Unable to load sharpclaw1 solver, did you run make?'
                pass
            solver.lim_type = 2             # WENO reconstruction
            solver.char_decomp = 0          # characteristic-wise reconstruction
            solver.char_bc = 0
    else:
        solver = pyclaw.ClawSolver1D(rs)

    solver.kernel_language = kernel_language

    solver.bc_lower[0]=pyclaw.BC.extrap
    solver.bc_upper[0]=pyclaw.BC.extrap

    mx = 400;
    # x = pyclaw.Dimension(-5.0,5.0,mx,name='x')
    x = pyclaw.Dimension(-1.,1.,mx,name='x')
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain,num_eqn)

    state.problem_data['gamma']= gamma

    if kernel_language =='Python':
        state.problem_data['efix'] = False

    xc = state.grid.p_centers[0]
    epsilon = 0.2
    # velocity = (xc<-4.)*2.629369
    # pressure = (xc<-4.)*10.33333 + (xc>=-4.)*1.

    beta = 5.; x0 = 0.; x1 = 0.
    state.q[density,:] = 1. + epsilon*np.sin(2.*np.pi*xc)
    velocity = 1.*np.ones(len(xc))
    pressure = 2.*np.ones(len(xc))

    # state.q[density ,:] = (xc<-4.)*3.857143 + (xc>=-4.)*(1+epsilon*np.sin(5*xc))
    state.q[momentum,:] = velocity * state.q[density,:]
    state.q[energy  ,:] = pressure/(gamma - 1.) + 0.5 * state.q[density,:] * velocity**2

    # initialize boundary cells if boundary condition is outflow
    if solver.bc_upper[0] == pyclaw.BC.outflow:
        ng = solver.num_ghost
        dx = state.grid.delta[0]
        xbc = np.linspace(xc[-1]+dx,xc[-1]+ng*dx,ng)
        state.qbc[density ,-ng:] = 1. + epsilon*np.sin(2*np.pi*xbc)
        state.qbc[momentum,-ng:] = velocity * state.qbc[density,-ng:]
        state.qbc[energy  ,-ng:] = pressure/(gamma - 1.) + 0.5 * state.qbc[density,-ng:] * velocity**2

        # print 'setup'
        # print state.qbc[:,-4:]
        # print

    claw = pyclaw.Controller()
    claw.tfinal = 2.
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
    claw.num_output_times = 10
    claw.outdir = outdir
    claw.setplot = setplot
    claw.keep_copy = True

    return claw

#--------------------------
def setplot(plotdata):
#--------------------------
    """ 
    Specify what is to be plotted at each frame.
    Input:  plotdata, an instance of visclaw.data.ClawPlotData.
    Output: a modified version of plotdata.
    """ 
    plotdata.clearfigures()  # clear any old figures,axes,items data

    # Figure for density
    plotfigure = plotdata.new_plotfigure(name='', figno=0)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.axescmd = 'subplot(311)'
    plotaxes.title = 'Density'
    # plotaxes.xlimits = (-5.,5.)
    plotaxes.xlimits = (-1.,1.)

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = density
    plotitem.kwargs = {'linewidth':3}
    
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'Momentum'
    plotaxes.axescmd = 'subplot(312)'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = momentum
    plotitem.kwargs = {'linewidth':3}
    # plotaxes.xlimits = (-5.,5.)
    plotaxes.xlimits = (-1.,1.)

    plotaxes = plotfigure.new_plotaxes()
    plotaxes.title = 'Energy'
    plotaxes.axescmd = 'subplot(313)'

    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = energy
    plotitem.kwargs = {'linewidth':3}
    # plotaxes.xlimits = (-5.,5.)
    plotaxes.xlimits = (-1.,1.)

    return plotdata

if __name__=="__main__":
    from nodepy import rk
    RK = rk.loadRKM('RK44')
    r, v, alphaup, alphadown = RK.optimal_perturbed_splitting()
    v = v.astype(float)
    alphaup = alphaup.astype(float)
    alphadown = alphadown.astype(float)
    RK.c = RK.c.astype(float)
    method_coeff = [r, v, alphaup, alphadown, RK.c]

    from clawpack.pyclaw.util import run_app_from_main
    output = run_app_from_main(setup,setplot)

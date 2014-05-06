#!/usr/bin/env python
# encoding: utf-8

r"""
Burgers' equation
=========================

Solve the inviscid Burgers' equation:

.. math:: 
    q_t + \frac{1}{2} (q^2)_x & = 0.

This is a nonlinear PDE often used as a very simple
model for fluid dynamics.

The initial condition is sinusoidal, but after a short time a shock forms
(due to the nonlinearity).
"""
alpha = [1./4.,0.,3./4.]
beta = [0.,0.,3./2.]
cfl_desired = 0.15
cfl_max = 0.2

def setup(use_petsc=0,kernel_language='Fortran',outdir='./_output',solver_type='sharpclaw',
        weno_order=5, lim_type=2, time_integrator='SSP104'):
    """
    Example python script for solving the 1d Burgers equation.
    """

    import numpy as np
    from clawpack import riemann

    if use_petsc:
        import clawpack.petclaw as pyclaw
    else:
        from clawpack import pyclaw

    #===========================================================================
    # Setup solver and solver parameters
    #===========================================================================
    if solver_type=='sharpclaw':
        if kernel_language=='Python': 
            solver = pyclaw.SharpClawSolver1D(riemann.burgers_1D_py.burgers_1D)
        elif kernel_language=='Fortran':
            solver = pyclaw.SharpClawSolver1D(riemann.burgers_1D)
        solver.weno_order=weno_order
        solver.time_integrator=time_integrator
        solver.lim_type=lim_type
        if time_integrator == 'LMM' or time_integrator == 'SSPMS32':
            solver.alpha = alpha
            solver.beta = beta
            solver.cfl_desired = cfl_desired
            solver.cfl_max = cfl_max
            solver.dt_initial = 0.0001
            solver.dt_variable = False
    else:
        if kernel_language=='Python': 
            solver = pyclaw.ClawSolver1D(riemann.burgers_1D_py.burgers_1D)
        elif kernel_language=='Fortran':
            solver = pyclaw.ClawSolver1D(riemann.burgers_1D)
        solver.limiters = pyclaw.limiters.tvd.vanleer

    solver.kernel_language = kernel_language
        
    solver.bc_lower[0] = pyclaw.BC.periodic
    solver.bc_upper[0] = pyclaw.BC.periodic

    #===========================================================================
    # Initialize domain and then initialize the solution associated to the domain
    #===========================================================================
    x = pyclaw.Dimension('x',0.0,1.0,500)
    domain = pyclaw.Domain(x)
    num_eqn = 1
    state = pyclaw.State(domain,num_eqn)

    grid = state.grid
    xc=grid.x.centers
    state.q[0,:] = np.sin(np.pi*2*xc) + 0.50
    state.problem_data['efix']=True

    #===========================================================================
    # Setup controller and controller parameters. Then solve the problem
    #===========================================================================
    claw = pyclaw.Controller()
    claw.tfinal =0.5
    claw.solution = pyclaw.Solution(state,domain)
    claw.solver = solver
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

    # Figure for q[0]
    plotfigure = plotdata.new_plotfigure(name='q[0]', figno=0)

    # Set up for axes in this figure:
    plotaxes = plotfigure.new_plotaxes()
    plotaxes.xlimits = 'auto'
    plotaxes.ylimits = [-1., 2.]
    plotaxes.title = 'q[0]'

    # Set up for item on these axes:
    plotitem = plotaxes.new_plotitem(plot_type='1d')
    plotitem.plot_var = 0
    plotitem.plotstyle = '-o'
    plotitem.color = 'b'
    
    return plotdata


if __name__=="__main__":
    from clawpack.pyclaw.util import run_app_from_main
    claw = run_app_from_main(setup,setplot)

    import numpy as np
    qfinal=claw.frames[claw.num_output_times].state.get_q_global()
    print np.linalg.norm(qfinal)

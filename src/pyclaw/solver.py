r"""
Module specifying the interface to every solver in PyClaw.
"""
import logging
import numpy as np

class CFLError(Exception):
    """Error raised when cfl_max is exceeded.  Is this a
       reasonable mechanism for handling that?"""
    def __init__(self,msg):
        super(CFLError,self).__init__(msg)

class BC():
    """Enumeration of boundary condition names."""
    # This could instead just be implemented as a static dictionary.
    custom = 0
    extrap = 1
    periodic = 2
    wall = 3
    outflow = 4

#################### Dummy routines ######################
def default_compute_gauge_values(q,aux):
    r"""By default, record values of q at gauges.
    """
    return q

def before_step(solver,solution):
    r"""
    Dummy routine called before each step
    
    Replace this routine if you want to do something before each time step.
    """
    pass

class Solver(object):
    r"""
    Pyclaw solver superclass.

    The pyclaw.Solver.solver class is an abstract class that should
    not be instantiated; rather, all Solver classes should inherit from it.

    Solver initialization takes one argument -- a Riemann solver:

        >>> from clawpack import pyclaw, riemann
        >>> solver = pyclaw.ClawSolver2D(riemann.euler_4wave_2D)

    after which solver options may be set.
    It is necessary to set the boundary conditions (for q, and
    for aux if an aux array is used):

        >>> solver.bc_lower[0] = pyclaw.BC.extrap
        >>> solver.bc_upper[0] = pyclaw.BC.wall
    
    Many other options may be set
    for specific solvers; for instance the limiter to be used, whether to
    use a dimensionally-split algorithm, and so forth.

    Usually the solver is attached to a controller before being used::

        >>> claw = pyclaw.Controller()
        >>> claw.solver = solver

    .. attribute:: dt
        
        Current time step, ``default = 0.1``
        
    .. attribute:: cfl
        
        Current Courant-Freidrichs-Lewy number, ``default = 1.0``
    
    .. attribute:: status
        
        Dictionary of status values for the solver with the following keys:
         - ``cflmax`` = Maximum CFL number
         - ``dtmin`` = Minimum time step taken
         - ``dtmax`` = Maximum time step taken
         - ``numsteps`` = Total number of time steps that have been taken

        solver.status is returned by solver.evolve_to_time.

    .. attribute:: before_step
    
        Function called before each time step is taken.
        The required signature for this function is:
        
        def before_step(solver,solution)

    .. attribute:: dt_variable
    
        Whether to allow the time step to vary, ``default = True``.
        If false, the initial time step size is used for all steps.
        
    .. attribute:: max_steps
    
        The maximum number of time steps allowd to reach the end time 
        requested, ``default = 10000``.  If exceeded, an exception is
        raised.
    
    .. attribute:: logger
    
        Default logger for all solvers.  Records information about the run
        and debugging messages (if requested).

    .. attribute:: bc_lower 
    
        (list of ints) Lower boundary condition types, listed in the
        same order as the Dimensions of the Patch.  See Solver.BC for
        an enumeration.

    .. attribute:: bc_upper 
    
        (list of ints) Upper boundary condition types, listed in the
        same order as the Dimensions of the Patch.  See Solver.BC for
        an enumeration.

    .. attribute:: user_bc_lower 
        
        (func) User defined lower boundary condition.
        Fills the values of qbc with the correct boundary values.
        The appropriate signature is:

        def user_bc_lower(patch,dim,t,qbc,num_ghost):

    .. attribute:: user_bc_upper 
    
        (func) User defined upper boundary condition.
        Fills the values of qbc with the correct boundary values.
        The appropriate signature is:

        def user_bc_upper(patch,dim,t,qbc,num_ghost):
 
        
    :Initialization:
    
    Output:
     - (:class:`Solver`) - Initialized Solver object
    """

            
    def __setattr__(self, key, value):
        if not hasattr(self, '_isinitialized'):
            self.__dict__['_isinitialized'] = False
        if self._isinitialized and not hasattr(self, key):
            raise TypeError("%s has no attribute %s" % (self.__class__,key))
        object.__setattr__(self,key,value)

    @property
    def all_bcs(self):
        return self.bc_lower, self.bc_upper
    @all_bcs.setter
    def all_bcs(self,all_bcs):
        for i in range(self.num_dim):
            self.bc_lower[i] = all_bcs
            self.bc_upper[i] = all_bcs


    #  ======================================================================
    #   Initialization routines
    #  ======================================================================
    def __init__(self,riemann_solver=None,claw_package=None):
        r"""
        Initialize a Solver object
        
        See :class:`Solver` for full documentation
        """ 
        # Setup solve logger
        self.logger = logging.getLogger('pyclaw.solver')

        self.dt_initial = 0.1
        self.dt_max = 1e99
        self.max_steps = 10000
        self.dt_variable = True
        self.num_waves = None #Must be set later to agree with Riemann solver
        self.qbc = None
        self.auxbc = None
        self.rp = None
        self.fmod = None
        self._is_set_up = False
        self._use_old_bc_sig = False
        self.accept_step = True
        self.before_step = None
        self.Fbc = 0.
        # self.told = None
        # self.qold = None

        # select package to build solver objects from, by default this will be
        # the package that contains the module implementing the derived class
        # for example, if ClawSolver1D is implemented in 'clawpack.petclaw.solver', then 
        # the computed claw_package will be 'clawpack.petclaw'
        
        import sys
        if claw_package is not None and claw_package in sys.modules:
            self.claw_package = sys.modules[claw_package]
        else:
            def get_clawpack_dot_xxx(modname): return modname.rpartition('.')[0].rpartition('.')[0]
            claw_package_name = get_clawpack_dot_xxx(self.__module__)
            if claw_package_name in sys.modules:
                self.claw_package = sys.modules[claw_package_name]
            else:
                raise NotImplementedError("Unable to determine solver package, please provide one")

        # Initialize time stepper values
        self.dt = self.dt_initial
        self.cfl = self.claw_package.CFL(self.cfl_desired)
       
        # Status Dictionary
        self.status = {'cflmax': -np.inf,
                       'dtmin': np.inf,
                       'dtmax': -np.inf,
                       'numsteps': 0 }
        
        # No default BCs; user must set them
        self.bc_lower =    [None]*self.num_dim
        self.bc_upper =    [None]*self.num_dim
        self.aux_bc_lower = [None]*self.num_dim
        self.aux_bc_upper = [None]*self.num_dim
        
        self.user_bc_lower = None
        self.user_bc_upper = None

        self.user_aux_bc_lower = None
        self.user_aux_bc_upper = None

        self.num_eqn   = None
        self.num_waves = None

        self.compute_gauge_values = default_compute_gauge_values
        r"""(function) - Function that computes quantities to be recorded at gauges"""

        self.qbc          = None
        r""" Array to hold ghost cell values.  This is the one that gets passed
        to the Fortran code.  """

        if riemann_solver is not None:
            self.rp = riemann_solver
            rp_name = riemann_solver.__name__.split('.')[-1]
            from clawpack import riemann
            if "ptwise" in rp_name:
                rp_name = rp_name.replace("_ptwise", "")
            self.num_eqn   = riemann.static.num_eqn.get(rp_name,None)
            self.num_waves = riemann.static.num_waves.get(rp_name,None)

        self._isinitialized = True

        super(Solver,self).__init__()


    # ========================================================================
    #  Solver setup and validation routines
    # ========================================================================
    def is_valid(self):
        r"""
        Checks that all required solver attributes are set.
        
        Checks to make sure that all the required attributes for the solver 
        have been set correctly.  All required attributes that need to be set 
        are contained in the attributes list of the class.
        
        Will post debug level logging message of which required attributes 
        have not been set.
        
        :Output:
         - *valid* - (bool) True if the solver is valid, False otherwise
        
        """
        valid = True
        reason = None
        if any([bcmeth == BC.custom for bcmeth in self.bc_lower]):
            if self.user_bc_lower is None:
                valid = False
                reason = 'Lower custom BC function has not been set.'
        if any([bcmeth == BC.custom for bcmeth in self.bc_upper]):
            if self.user_bc_upper is None:
                valid = False
                reason = 'Upper custom BC function has not been set.'
        if self.num_waves is None:
            valid = False
            reason = 'solver.num_waves has not been set.'
        if self.num_eqn is None:
            valid = False
            reason = 'solver.num_eqn has not been set.'
        if (None in self.bc_lower) or (None in self.bc_upper):
            valid = False
            reason = 'One of the boundary conditions has not been set.'


        if reason is not None:
            self.logger.debug(reason)
        return valid, reason
        
    def setup(self,solution):
        r"""
        Stub for solver setup routines.
        
        This function is called before a set of time steps are taken in order 
        to reach tend.  A subclass should extend or override it if it needs to 
        perform some setup based on attributes that would be set after the 
        initialization routine.  Typically this is initialization that
        requires knowledge of the solution object.
        """

        self._is_set_up = True

    def __del__(self):
        r"""
        Stub for solver teardown routines.
        
        This function is called at the end of a simulation.
        A subclass should override it only if it needs to 
        perform some cleanup, such as deallocating arrays in a Fortran module.
        """
        self._is_set_up = False



    def __str__(self):
        output = "Solver Status:\n"
        for (k,v) in self.status.iteritems():
            output = "\n".join((output,"%s = %s" % (k.rjust(25),v)))
        return output


    # ========================================================================
    #  Boundary Conditions
    # ========================================================================    
    def _allocate_bc_arrays(self,state):
        r"""
        Create numpy arrays for q and aux with ghost cells attached.
        These arrays are referred to throughout the code as qbc and auxbc.

        This is typically called by solver.setup().
        """
        import inspect
        for fun in (self.user_bc_lower,self.user_bc_upper,self.user_aux_bc_lower,self.user_aux_bc_upper):
            if fun is not None:
                args = inspect.getargspec(fun)[0]
                if len(args) == 5:
                    self.logger.warn("""The custom boundary condition
                                        function signature has been changed.
                                        The previous signature will not be
                                        supported in Clawpack 6.0.  Please see 
                                        http://www.clawpack.org/pyclaw/solvers.html#change-to-custom-bc-function-signatures
                                        for more information.""")
                    self._use_old_bc_sig = True



        qbc_dim = [n+2*self.num_ghost for n in state.grid.num_cells]
        qbc_dim.insert(0,state.num_eqn)
        self.qbc = np.zeros(qbc_dim,order='F')

        auxbc_dim = [n+2*self.num_ghost for n in state.grid.num_cells]
        auxbc_dim.insert(0,state.num_aux)
        self.auxbc = np.empty(auxbc_dim,order='F')
        
        self._apply_bcs(state)


    def _apply_bcs(self, state):
        r"""
        Apply boundary conditions to both q and aux arrays.
        
        In the case of a user-defined boundary condition, both arrays
        qbc and auxbc are passed to the user function.  Typically the
        function would only modify one or the other of them, though this
        is not enforced.

        If the user function only accepts one array argument, we warn
        that this interface has been deprecated.  In Clawpack 6, we will
        drop backward compatibility.

        For parallel runs, we check whether we're actually on a domain
        boundary.  If we are just at an inter-patch boundary, nothing needs to
        be done here.
        """

        import numpy as np

        self.qbc = state.get_qbc_from_q(self.num_ghost, self.qbc)
        if state.num_aux > 0:
            self.auxbc = state.get_auxbc_from_aux(self.num_ghost, self.auxbc)
        
        grid = state.grid

        for (idim, dim) in enumerate(grid.dimensions):
            # Check if we are on a true boundary
            if state.grid.on_lower_boundary[idim]:

                bcs = []
                if state.num_aux > 0:
                    bcs.append({'array'  : self.auxbc,
                                'type'   : self.aux_bc_lower,
                                'custom_fun' : self.user_aux_bc_lower,
                                'variable' : 'aux'})
                bcs.append({'array'  : self.qbc,
                            'type'   : self.bc_lower,
                            'custom_fun' : self.user_bc_lower,
                            'variable' : 'q'})
                for (i, bc) in enumerate(bcs):

                    if bc['type'][idim] == BC.custom:
                        if not self._use_old_bc_sig: 
                            bc['custom_fun'](state, dim, state.t, self.qbc, self.auxbc,
                                      self.num_ghost)
                        else:
                            bc['custom_fun'](state, dim, state.t, bc['array'], self.num_ghost)
                    
                    elif bc['type'][idim] == BC.periodic \
                            and not state.grid.on_upper_boundary[idim]:
                        pass # In a parallel run, # PETSc handles periodic BCs.

                    else:
                        self._bc_lower(bc['type'][idim], state, dim, state.t,
                                        np.rollaxis(bc['array'], idim+1, 1), idim,
                                        bc['variable'])

            if state.grid.on_upper_boundary[idim]:

                bcs = []
                if state.num_aux > 0:
                    bcs.append({'array'  : self.auxbc,
                                'type'   : self.aux_bc_upper,
                                'custom_fun' : self.user_aux_bc_upper,
                                'variable' : 'aux'})
                bcs.append({'array'  : self.qbc,
                            'type'   : self.bc_upper,
                            'custom_fun' : self.user_bc_upper,
                            'variable' : 'q'})
                for (i, bc) in enumerate(bcs):

                    if bc['type'][idim] == BC.custom:
                        if not self._use_old_bc_sig: 
                            bc['custom_fun'](state, dim, state.t, self.qbc, self.auxbc,
                                      self.num_ghost)
                        else:
                            bc['custom_fun'](state, dim, state.t, bc['array'], self.num_ghost)
                    
                    elif bc['type'][idim] == BC.periodic \
                            and not state.grid.on_lower_boundary[idim]:
                        pass # In a parallel run, # PETSc handles periodic BCs.

                    else:
                        self._bc_upper(bc['type'][idim], state, dim, state.t,
                                        np.rollaxis(bc['array'], idim+1, 1), idim,
                                        bc['variable'])


    def _bc_lower(self, bc_type, state, dim, t, array, idim, name):
        r"""
        Apply lower boundary conditions to array.
        
        Sets the lower coordinate's ghost cells of *array* depending on what 
        :attr:`bc_lower` is.  If :attr:`bc_lower` = 0 then the user 
        boundary condition specified by :attr:`user_bc_lower` is used.  Note 
        that in this case the function :attr:`user_bc_lower` belongs only to 
        this dimension but :attr:`user_bc_lower` could set all user boundary 
        conditions at once with the appropriate calling sequence.
        
        :Input:
         - *patch* - (:class:`Patch`) Patch that the dimension belongs to.
         
        :Input/Ouput:
         - *array* - (ndarray(...,num_eqn)) Array with added ghost cells which 
           will be set in this routines.
        """

        if bc_type == BC.extrap:
            for i in xrange(self.num_ghost):
                array[:,i,...] = array[:,self.num_ghost,...]
        elif bc_type == BC.outflow:
            for i in xrange(self.num_ghost):
                # array[:,self.num_ghost-1-i,...] = 2.*array[:,self.num_ghost-i,...] - array[:,self.num_ghost+1-i,...]
                # array[:,self.num_ghost-1-i,...] = 2.*array[:,self.num_ghost,...] - array[:,self.num_ghost+1,...]
                array[:,self.num_ghost-1-i,...] = 5.*array[:,self.num_ghost-i,...] - 10.*array[:,self.num_ghost+1-i,...] + \
                    10.*array[:,self.num_ghost+2-i,...] - 5.*array[:,self.num_ghost+3-i,...] + array[:,self.num_ghost+4-i,...]
        elif bc_type == BC.periodic:
            # This process owns the whole patch
            array[:,:self.num_ghost,...] = array[:,-2*self.num_ghost:-self.num_ghost,...]
        elif bc_type == BC.wall:
            if name == 'q':
                for i in xrange(self.num_ghost):
                    array[:,i,...] = array[:,2*self.num_ghost-1-i,...]
                    array[self.reflect_index[idim],i,...] = -array[self.reflect_index[idim],2*self.num_ghost-1-i,...] # Negate normal velocity
            else:
                for i in xrange(self.num_ghost):
                    array[:,i,...] = array[:,2*self.num_ghost-1-i,...]
        else:
            raise NotImplementedError("Boundary condition %s not implemented" % bc_type)


    def _bc_upper(self, bc_type, state, dim, t, array, idim, name):
        r"""
        Apply upper boundary conditions to array
        
        Sets the upper coordinate's ghost cells of *array* depending on what 
        :attr:`bc_upper` is.  If :attr:`bc_upper` = 0 then the user 
        boundary condition specified by :attr:`user_bc_upper` is used.  Note 
        that in this case the function :attr:`user_bc_upper` belongs only to 
        this dimension but :attr:`user_bc_upper` could set all user boundary 
        conditions at once with the appropriate calling sequence.
        
        :Input:
         - *patch* - (:class:`Patch`) Patch that the dimension belongs to
         
        :Input/Ouput:
         - *array* - (ndarray(...,num_eqn)) Array with added ghost cells which will
           be set in this routines
        """

        if bc_type == BC.extrap:
            for i in xrange(self.num_ghost):
                array[:,-i-1,...] = array[:,-self.num_ghost-1,...]
        elif bc_type == BC.outflow:

            self._weno_type_recon2(state, array, idim)

            # array[:,-self.num_ghost,...] = 5.*array[:,-self.num_ghost-1,...] - 10.*array[:,-self.num_ghost-2,...] + \
            #         10.*array[:,-self.num_ghost-3,...] - 5.*array[:,-self.num_ghost-4,...] + array[:,-self.num_ghost-5,...]
            # array[:,-self.num_ghost+1,...] = 15.*array[:,-self.num_ghost-1,...] - 40.*array[:,-self.num_ghost-2,...] + \
            #         45.*array[:,-self.num_ghost-3,...] - 24.*array[:,-self.num_ghost-4,...] + 5.*array[:,-self.num_ghost-5,...]
            # array[:,-self.num_ghost+2,...] = 35.*array[:,-self.num_ghost-1,...] - 105.*array[:,-self.num_ghost-2,...] + \
            #         126.*array[:,-self.num_ghost-3,...] - 70.*array[:,-self.num_ghost-4,...] + 15.*array[:,-self.num_ghost-5,...]

            # for i in xrange(self.num_ghost):
                # array[:,-self.num_ghost+i,...] = 0.
            
            # for i in xrange(self.num_ghost):
                # array[:,-self.num_ghost+i,...] = 2.*array[:,-self.num_ghost-1+i,...] - array[:,-self.num_ghost-2+i,...]
            #     array[:,-self.num_ghost+i,...] = 2.*array[:,-self.num_ghost-1,...] - array[:,-self.num_ghost-2,...]
                # array[:,-self.num_ghost+i,...] = 5.*array[:,-self.num_ghost-1+i,...] - 10.*array[:,-self.num_ghost-2+i,...] + \
                    # 10.*array[:,-self.num_ghost-3+i,...] - 5.*array[:,-self.num_ghost-4+i,...] + array[:,-self.num_ghost-5+i,...]
        elif bc_type == BC.periodic:
            # This process owns the whole patch
            array[:,-self.num_ghost:,...] = array[:,self.num_ghost:2*self.num_ghost,...]
        elif bc_type == BC.wall:
            if name == 'q':
                for i in xrange(self.num_ghost):
                    array[:,-i-1,...] = array[:,-2*self.num_ghost+i,...]
                    array[self.reflect_index[idim],-i-1,...] = -array[self.reflect_index[idim],-2*self.num_ghost+i,...] # Negate normal velocity
            else:
                for i in xrange(self.num_ghost):
                    array[:,-i-1,...] = array[:,-2*self.num_ghost+i,...]
        else:
            raise NotImplementedError("Boundary condition %s not implemented" % self.bc_lower)


    def _weno_type_recon2(self, state, array, idim):
        r"""
        Apply upper boundary conditions to array's ghost cells based on a WEN0-type
        reconstruction
        
        Sets the upper coordinate's ghost cells of *array* depending on what 
        :attr:`bc_upper` is.  If :attr:`bc_upper` = 0 then the user 
        boundary condition specified by :attr:`user_bc_upper` is used.  Note 
        that in this case the function :attr:`user_bc_upper` belongs only to 
        this dimension but :attr:`user_bc_upper` could set all user boundary 
        conditions at once with the appropriate calling sequence.
        
        :Input:
         - *array* - (ndarray(...,num_eqn)) Array with added ghost cells which will
           be set in this routine

        :Input:
         - *ghost_cells* - (ndarray(...,num_eqn)) Array with ghost cell's values
            added by a WENO-type reconstruction
        """

        # # do nothing at time = 0 because boundary cells are already filled from initial conditions
        # if state.t == 0.:
        #     self.told = state.t
        #     self.qold = np.zeros(array.shape)
        #     self.qold[:] = array
        #     return

        # Fill chost cells
        for i in xrange(self.num_ghost):
            array[:,-i-1,...] = array[:,-self.num_ghost-1,...]#state.qbc[:,-i-1]
        
        num_eqn = array.shape[0]
        # dt = state.t - self.told
        dx = state.grid.delta[idim]

        # Find primitive variables
        gamma = state.problem_data['gamma']
        gamma1 = state.problem_data['gamma'] - 1.
        rho = array[0,-self.num_ghost-10:,...]
        u = array[1,-self.num_ghost-10:,...]/array[0,-self.num_ghost-10:,...]
        p = gamma1*(array[2,-self.num_ghost-10:,...] - 0.5*rho*u**2)
        prim_var = np.vstack([rho,u,p])

        xc = state.grid.p_centers[0]
        xbc = np.linspace(xc[-10],xc[-1]+self.num_ghost*dx,self.num_ghost+10)
        rho_ref = 1. + 0.2*np.sin(2*np.pi*(xbc - state.t))
        print '|rho - rho_ref|'
        print np.abs(rho - rho_ref)
        print 'rho,u,p'
        print rho
        print u
        print p 
        print

        # Approximate spatial derivatives of primitive variables with one-side 3-point differences
        Dprim_var = (3.*prim_var[:,-self.num_ghost:] - 4.*prim_var[:,-self.num_ghost-1:-1] + prim_var[:,-self.num_ghost-2:-2])/(2.*dx)

        # print 'Dprim_var'
        # print Dprim_var
        # print

        # Compute amplitude variations of characteristic waves (Li's)
        rho = rho[-self.num_ghost:]
        u = u[-self.num_ghost:]
        p = p[-self.num_ghost:]
        c = np.sqrt(gamma*p/rho)
        prim_var = np.vstack([rho,u,p])

        # print 'prim_var - old ghost cells'
        # print prim_var
        # print

        L1 = .0#(u-c)*(Dprim_var[2,:] - rho*c*Dprim_var[1,:])
        L2 =     u*(c**2*Dprim_var[0,:] - rho*c*Dprim_var[2,:])
        L3 = (u+c)*(Dprim_var[2,:] + rho*c*Dprim_var[1,:])

        # print 'Li'
        # print L1
        # print L2
        # print L3
        # print

        # solve LODI system
        self.Fbc = np.zeros(state.qbc.shape)
        d1 = (L2 + 0.5*(L3 + L1))/c**2
        d2 = 0.5*(L3 + L1)
        d3 = (L3 - L1)/(2.*rho*c)
        self.Fbc[0,-self.num_ghost:] = - d1
        self.Fbc[1,-self.num_ghost:] = - (u*d1 + rho*d3)
        self.Fbc[2,-self.num_ghost:] = - (0.5*u**2*d1 + d2/gamma1 + rho*u*d3)

        # print 'RHS'
        # print self.Fbc[:,-self.num_ghost:]
        # print

        # self.qold[:,-self.num_ghost:,...] = array[:,-self.num_ghost:,...]
        # self.told = state.t

        # print 'ghost cells'
        # print array[:,-self.num_ghost:,...]
        # print


    def _weno_type_recon(self, state, array, idim):
        r"""
        Apply upper boundary conditions to array's ghost cells based on a WEN0-type
        reconstruction
        
        Sets the upper coordinate's ghost cells of *array* depending on what 
        :attr:`bc_upper` is.  If :attr:`bc_upper` = 0 then the user 
        boundary condition specified by :attr:`user_bc_upper` is used.  Note 
        that in this case the function :attr:`user_bc_upper` belongs only to 
        this dimension but :attr:`user_bc_upper` could set all user boundary 
        conditions at once with the appropriate calling sequence.
        
        :Input:
         - *array* - (ndarray(...,num_eqn)) Array with added ghost cells which will
           be set in this routine

        :Input:
         - *ghost_cells* - (ndarray(...,num_eqn)) Array with ghost cell's values
            added by a WENO-type reconstruction
        """

        mx = state.grid.num_cells[0]
        num_eqn = array.shape[0]
        
        evr = np.zeros((num_eqn,num_eqn,5))
        evl = np.zeros((num_eqn,num_eqn,5))
        rp_name = self.rp.__name__.split('.')[-1]
        if rp_name == 'acoustics_1D':
            zz = state.problem_data['zz']
            cc = state.problem_data['cc']
            # e-values at boundary
            evalues = [-cc, cc]
            # left and right eigenvectors
            for i in xrange(5):
                evr[:,:,i] = np.array([[-zz,zz],[1,1]])
                evl[:,:,i] = np.array([[-0.5/zz,0.5],[0.5/zz,0.5]])
            # evl,evr = self.fmod.evec(mx,self.num_ghost,mx,array,array,array)
        elif rp_name == 'euler_with_efix_1D':
            # e-values at boundary
            # Rinv,R = self.fmod.evec(mx,self.num_ghost,mx,array,array,array)
            # evalues = R[1,:,-self.num_ghost]

            # left and right eigenvectors:
            gamma = state.problem_data['gamma']
            gamma1 = gamma-1.
            for i in xrange(5):
                rho = array[0,-self.num_ghost-5+i,...]
                u = array[1,-self.num_ghost-5+i,...]/array[0,-self.num_ghost-5+i,...]
                E = array[2,-self.num_ghost-5+i,...]
                p = gamma1*(E - 0.5*rho*u**2)
                H = (E + p)/rho
                c = np.sqrt(gamma*p/rho)

                evr[:,:,i] = np.array([[1.,    1.,       1.   ],
                                       [u-c,   u,        u+c  ],
                                       [H-u*c, 0.5*u**2, H+u*c]])

                evl[:,:,i] = gamma1/(2.*c**2)* \
                    np.array([[u*c/gamma1+0.5*u**2,  -c/gamma1-u,  1. ],
                              [2.*(H-u**2),          2.*u,         -2.],
                              [-u*c/gamma1+0.5*u**2, c/gamma1 - u, 1. ]])

            evalues = evr[1,:,-1]

        # Project interior solution values to the characteristic field
        w = np.zeros((num_eqn,5+self.num_ghost))
        for i in xrange(5):
            w[:,i] = np.dot(evl[:,:,i],array[:,-self.num_ghost-5+i,...])

        # WENO-type reconstruction 
        dx = state.grid.delta[idim]
        epsilon = 1.e-6
        beta = np.zeros((num_eqn,5))
        tomega = np.zeros((num_eqn,5))
        omega = np.zeros((num_eqn,5))

        qN   = w[:,4]
        qNm1 = w[:,3]
        qNm2 = w[:,2]
        qNm3 = w[:,1]
        qNm4 = w[:,0]

        # Smoothness indicators
        beta[:,0] = dx**2
        beta[:,1] = (-qNm1+qN)**2
        beta[:,2] = 1./12.*(25.*qNm2**2 + 160.*qNm1**2 + 61.*qN**2 - 196.*qNm1*qN + qNm2*(-124.*qNm1+74.*qN))
        beta[:,3] = 1./180.*(814.*qNm3**2 + 10536.*qNm2**2 + 16476.*qNm1**2 + 3034.*qN**2 - 13989.*qNm1*qN \
            - 3.*qNm2*(8699.*qNm1-3618.*qN) + qNm3*(-5829.*qNm2+7134.*qNm1-2933.*qN))
        beta[:,4] = 1./30240.* \
            (304207.*qNm4**2 + 6491032.*qNm3**2 + 20495952.*qNm2**2 + 13782232.*qNm1**2 + 1426657.*qN**2 \
            - 33389628.*qNm2*qNm1 + 10446126.*qNm2*qN - 8771498.*qNm1*qN \
            + qNm4*(-2805398.*qNm3+4946226.*qNm2-3970394.*qNm1+1221152.*qN) \
            - 2.*qNm3*(11497314.*qNm2-9283528.*qNm1+2874547.*qN))

        # Nonlinear weights
        q = 3
        tomega[:,0] = dx**4/(epsilon + beta[:,0])**q
        tomega[:,1] = dx**3/(epsilon + beta[:,1])**q
        tomega[:,2] = dx**2/(epsilon + beta[:,2])**q
        tomega[:,3] = dx**1/(epsilon + beta[:,3])**q
        tomega[:,4] = (1. - dx - dx**2 - dx**3 - dx**4)/(epsilon + beta[:,4])**q

        somega = np.sum(tomega,1)
        omega[:,0] = tomega[:,0]/somega[:]
        omega[:,1] = tomega[:,1]/somega[:]
        omega[:,2] = tomega[:,2]/somega[:]
        omega[:,3] = tomega[:,3]/somega[:]
        omega[:,4] = tomega[:,4]/somega[:]

        d0wR = omega[:,0]*qN - omega[:,1]*(qNm1-3.*qN)/2. + omega[:,2]*(3.*qNm2-10.*qNm1+15.*qN)/8. + omega[:,3]*(-5.*qNm3+7.*(3.*qNm2-5.*qNm1+5.*qN))/16. + omega[:,4]*(35.*qNm4-180.*qNm3+378.*qNm2-420.*qNm1+315.*qN)/128.
        d1wR = omega[:,1]*(-qNm1+qN)/dx + omega[:,2]*(qNm2-3.*qNm1+2.*qN)/dx - omega[:,3]*(23.*qNm3-93.*qNm2+141.*qNm1-71.*qN)/(24.*dx) + omega[:,4]*(22.*qNm4-111.*qNm3+225.*qNm2-229.*qNm1+93.*qN)/(24.*dx)
        d2wR = omega[:,2]*(qNm2-2.*qNm1+qN)/(dx**2) - omega[:,3]*(3.*qNm3-11.*qNm2+13.*qNm1-5.*qN)/(2.*dx**2) + omega[:,4]*(43.*qNm4-208.*qNm3+390.*qNm2-328.*qNm1+103.*qN)/(24.*dx**2)
        d3wR = omega[:,3]*(-qNm3+3.*qNm2-3.*qNm1+qN)/(dx**3) + omega[:,4]*(2.*qNm4-9.*qNm3+15.*qNm2-11.*qNm1+3.*qN)/(dx**3)
        d4wR = omega[:,4]*(qNm4-4.*qNm3+6.*qNm2-4.*qNm1+qN)/(dx**4)

        r"""# CASES:

            1. Using exact solution at boundaries and finding spatial derivatives at boundary by using PDE.
            2. Tan-Shu approach - still this does not work, for some reason it generates an ingoing acoustic wave.
            3. LODI.
            4. Using WENO-type extrapolation for outgoing waves and setting incoming waves to zero.
        """

        case = 1

        if case == 1:
    ######### Using exact formulas at the boundary and using pde system to get 
            # spatial derivatives up to second order. This is like cheating since
            # information about solution at the boundary is not known. ###

            dx = state.grid.delta[idim]
            num_eqn = array.shape[0]
            d0qR = np.zeros(num_eqn)
            d1qR = np.zeros(num_eqn)
            d2qR = np.zeros(num_eqn)

            epsilon = 0.2
            g = 1. + epsilon*np.sin(2*np.pi*(1-state.t))
            gprime1 = -2*np.pi*epsilon*np.cos(2*np.pi*(1-state.t))
            gprime2 = 4*np.pi**2*epsilon*np.sin(2*np.pi*(1-state.t))
            gamma = state.problem_data['gamma']
            gamma1 = gamma-1.

            d0qR[0] = g
            d0qR[1] = g
            d0qR[2] = 2./gamma1 + 0.5*g
            
            d1qR[0] = -gprime1
            d1qR[1] = -gprime1
            d1qR[2] = -0.5*gprime1

            d2qR[0] = 2.*(gprime2 + 1./g)
            d2qR[1] = 2.*(gprime2 + 1./g)
            d2qR[2] = gprime2 + 1./g

            for i in xrange(self.num_ghost):
                array[:,-self.num_ghost+i,...] = d0qR + (i+0.5)*dx*d1qR + ((i+0.5)*dx)**2/2.*d2qR

            return

    #########

        elif case == 2:
    ######### Tan-Shu approach: Still this does not work ###

            R = evr[:,:,-1]
            d0qR = np.zeros(num_eqn)
            d1qR = np.zeros(num_eqn)
            d2qR = np.zeros(num_eqn)

            epsilon = 0.2
            g = 1. + epsilon*np.sin(2*np.pi*(1-state.t))
            gprime1 = -2*np.pi*epsilon*np.cos(2*np.pi*(1-state.t))
            gprime2 = 4*np.pi**2*epsilon*np.sin(2*np.pi*(1-state.t))

            d0qR[0] = g
            d0wR[0] = (-R[0,1]*d0wR[1] - R[0,2]*d0wR[2] + d0qR[0])/R[0,0]
            d0qR[1] = R[1,0]*d0wR[0] + R[1,1]*d0wR[1] + R[1,2]*d0wR[2]
            d0qR[2] = R[2,0]*d0wR[0] + R[2,1]*d0wR[1] + R[2,2]*d0wR[2]

            d1qR[1] = - gprime1
            d1wR[0] = (-R[1,1]*d1wR[1] - R[1,2]*d1wR[2] + d1qR[1])/R[1,0]
            d1qR[0] = R[0,0]*d1wR[0] + R[0,1]*d1wR[1] + R[0,2]*d1wR[2]
            d1qR[2] = R[2,0]*d1wR[0] + R[2,1]*d1wR[1] + R[2,2]*d1wR[2]

            for i in xrange(self.num_ghost):
                array[:,-self.num_ghost+i,...] = d0qR #+ (i+0.5)*dx*d1qR #+ ((i+0.5)*dx)**2/2.*d2qR \
                    # + ((i+0.5)*dx)**3/6.*d3qR + ((i+0.5)*dx)**4/24.*d4qR

            return

    #########

        elif case == 3:
    ######### LODI approach ###

            # Find primitive and coservative variables of the last cell average
            rho = array[0,-self.num_ghost-1,...]
            u = array[1,-self.num_ghost-1,...]/array[0,-self.num_ghost-1,...]
            E = array[2,-self.num_ghost-5+i,...]
            p = gamma1*(E - 0.5*rho*u**2)
            c = np.sqrt(gamma*p/rho)

            # Find Li's at boundary
            L1 = 0.
            L2 = u*d1wR[1]
            L3 = 0.#(u+c)*d1wR[2]

            # Find spatial derivatives of primitive variables at boudary
            rho_x = (L2/u + 0.5*(L3/(u+c) + L1/(u-c)))/c**2
            p_x = 0.5*(L3/(u+c) + L1/(u-c))
            u_x = (L3/(u+c) - L1/(u-c))/(2.*rho*c)

            # Find first two terms of the Taylor expansion of conservaive variables around the boundary
            num_eqn = array.shape[0]
            d0qR = np.zeros(num_eqn)
            d1qR = np.zeros(num_eqn)
            d2qR = np.zeros(num_eqn)

            epsilon = 0.2
            g = 1. + epsilon*np.sin(2*np.pi*(1-state.t))
            gprime1 = -2*np.pi*epsilon*np.cos(2*np.pi*(1-state.t))
            gprime2 = 4*np.pi**2*epsilon*np.sin(2*np.pi*(1-state.t))
            gamma = state.problem_data['gamma']
            gamma1 = gamma-1.

            d0qR[0] = g
            d0qR[1] = g
            d0qR[2] = 2./gamma1 + 0.5*g

            # Try to use values from the last cell since we don't know variables at boundary
            # d0qR[0] = array[0,-self.num_ghost-1,...] 
            # d0qR[1] = array[1,-self.num_ghost-1,...] 
            # d0qR[2] = array[2,-self.num_ghost-1,...] 

            # Use the derivatives of primitive variables to find derivatives for conservative variables
            d1qR[0] = rho_x
            d1qR[1] = rho*u_x + u*rho_x
            d1qR[2] = p_x/gamma1 + 0.5*u**2*rho_x + rho*u*u_x

            d2qR[0] = (gamma*u*p/rho**2*rho_x + d2wR[1])/(u*c**2)
            d2qR[1] = u*d2qR[0]
            d2qR[2] = 0.5*u**2*d2qR[0]

            # Fill ghost cells using Taylor expansion
            dx = state.grid.delta[idim]
            for i in xrange(self.num_ghost):
                array[:,-self.num_ghost+i,...] = d0qR + (i+0.5)*dx*d1qR + ((i+0.5)*dx)**2/2.*d2qR

            return

    #########

        elif case == 4:
    ######### WENO-type extrapolation ###

            # Direct computation of characteristic variables in ghost cells
            w[:,-self.num_ghost] = qNm2[:]*omega[:,2]-qNm3[:]*omega[:,3]+4.*qNm2[:]*omega[:,3]+qNm4[:]*omega[:,4]-5.*qNm3[:]*omega[:,4]+10.*qNm2[:]*omega[:,4] \
                +qN[:]*(1.+omega[:,1]+2.*omega[:,2]+3.*omega[:,3]+4.*omega[:,4])-qNm1[:]*(omega[:,1]+3.*omega[:,2]+6.*omega[:,3]+10.*omega[:,4])

            w[:,-self.num_ghost+1] = 3.*qNm2[:]*omega[:,2]-4.*qNm3[:]*omega[:,3]+15.*qNm2[:]*omega[:,3]+5.*qNm4[:]*omega[:,4]-24.*qNm3[:]*omega[:,4]+45.*qNm2[:]*omega[:,4] \
                +qN[:]*(1.+2.*omega[:,1]+5.*omega[:,2]+9.*omega[:,3]+14.*omega[:,4])-2.*qNm1[:]*(omega[:,1]+4.*omega[:,2]+10.*omega[:,3]+20.*omega[:,4])

            w[:,-self.num_ghost+3] = 6.*qNm2[:]*omega[:,2]-10.*qNm3[:]*omega[:,3]+36.*qNm2[:]*omega[:,3]+15.*qNm4[:]*omega[:,4]-70.*qNm3[:]*omega[:,4]+126.*qNm2[:]*omega[:,4] \
                +qN[:]*(1.+3.*omega[:,1]+9.*omega[:,2]+19.*omega[:,3]+34.*omega[:,4])-3.*qNm1[:]*(omega[:,1]+5.*(omega[:,2]+3.*omega[:,3]+7.*omega[:,4]))

            # Compute charactertic variables in ghost cells using Taylor expansion
            # for i in xrange(self.num_ghost):
            #     w[:,-self.num_ghost+i] = d0wR + (i+0.5)*dx*d1wR + ((i+0.5)*dx)**2/2.*d2wR \
            #         + ((i+0.5)*dx)**3/6.*d3wR + ((i+0.5)*dx)**4/24.*d4wR


            # Set ingoing wave to zero
            w[0,-self.num_ghost:] = 0.
            w[1,-self.num_ghost:] = 0.
            # w[0,-self.num_ghost:] = w[0,-self.num_ghost-1]

            # Project to physical space
            for i in xrange(self.num_ghost):
                array[:,-self.num_ghost+i,...] = np.dot(evr[:,:,-1],w[:,-self.num_ghost+i])

            return

    #########

        elif case == 5:
    ######### Directly project characteristic derivatives to physical space and fill ghost 
            # by using Taylor expansion. ###


            # Set ingoing component of wave derivatives to zero
            d0wR[0] = 0.
            d1wR[0] = 0.
            d2wR[0] = 0.
            d3wR[0] = 0.
            d4wR[0] = 0.

            # Project derivative approximations back to physical space
            d0qR = np.dot(evr[:,:,-1],d0wR)
            d1qR = np.dot(evr[:,:,-1],d1wR)
            d2qR = np.dot(evr[:,:,-1],d2wR)
            d3qR = np.dot(evr[:,:,-1],d3wR)
            d4qR = np.dot(evr[:,:,-1],d4wR)

            for i in xrange(self.num_ghost):
                array[:,-self.num_ghost+i,...] = d0qR + (i+0.5)*dx*d1qR + ((i+0.5)*dx)**2/2.*d2qR \
                    + ((i+0.5)*dx)**3/6.*d3qR + ((i+0.5)*dx)**4/24.*d4qR

            return

    #########


        # d0qR = qN
        # d1qR = omega[:,1]*(-qNm1+qN)/dx + omega[:,2]*(qNm2-4.*qNm1+3.*qN)/(2.*dx) + omega[:,3]*(-2.*qNm3+9.*qNm2-18.*qNm1+11.*qN)/(6.*dx) + omega[:,4]*(3.*qNm4-16.*qNm3+36.*qNm2-48.*qNm1+25.*qN)/(12.*dx)
        # d2qR = omega[:,2]*(qNm2-2.*qNm1+qN)/(dx**2) + omega[:,3]*(-qNm3+4.*qNm2-5.*qNm1+2.*qN)/(dx**2) + omega[:,4]*(11.*qNm4-56.*qNm3+114.*qNm2-104.*qNm1+35.*qN)/(12.*dx**2)
        # d3qR = omega[:,3]*(-qNm3+3.*qNm2-3.*qNm1+qN)/(dx**3) + omega[:,4]*(3.*qNm4-14.*qNm3+24.*qNm2-18.*qNm1+5.*qN)/(2.*dx**3)
        # d4qR = omega[:,4]*(qNm4-4.*qNm3+6.*qNm2-4.*qNm1+qN)/(dx**4)


        # Compute characteristic variables in ghost cells
        # w[:,-self.num_ghost] = qNm2[:]*omega[:,2]-qNm3[:]*omega[:,3]+4.*qNm2[:]*omega[:,3]+qNm4[:]*omega[:,4]-5.*qNm3[:]*omega[:,4]+10.*qNm2[:]*omega[:,4] \
        #     +qN[:]*(1.+omega[:,1]+2.*omega[:,2]+3.*omega[:,3]+4.*omega[:,4])-qNm1[:]*(omega[:,1]+3.*omega[:,2]+6.*omega[:,3]+10.*omega[:,4])

        # w[:,-self.num_ghost+1] = 3.*qNm2[:]*omega[:,2]-4.*qNm3[:]*omega[:,3]+15.*qNm2[:]*omega[:,3]+5.*qNm4[:]*omega[:,4]-24.*qNm3[:]*omega[:,4]+45.*qNm2[:]*omega[:,4] \
        #     +qN[:]*(1.+2.*omega[:,1]+5.*omega[:,2]+9.*omega[:,3]+14.*omega[:,4])-2.*qNm1[:]*(omega[:,1]+4.*omega[:,2]+10.*omega[:,3]+20.*omega[:,4])

        # w[:,-self.num_ghost+3] = 6.*qNm2[:]*omega[:,2]-10.*qNm3[:]*omega[:,3]+36.*qNm2[:]*omega[:,3]+15.*qNm4[:]*omega[:,4]-70.*qNm3[:]*omega[:,4]+126.*qNm2[:]*omega[:,4] \
        #     +qN[:]*(1.+3.*omega[:,1]+9.*omega[:,2]+19.*omega[:,3]+34.*omega[:,4])-3.*qNm1[:]*(omega[:,1]+5.*(omega[:,2]+3.*omega[:,3]+7.*omega[:,4]))

        # w[:,-self.num_ghost] = 5.*w[:,-self.num_ghost-1] - 10.*w[:,-self.num_ghost-2] + \
        #         10.*w[:,-self.num_ghost-3] - 5.*w[:,-self.num_ghost-4] + w[:,-self.num_ghost-5]
        # w[:,-self.num_ghost+1] = 15.*w[:,-self.num_ghost-1] - 40.*w[:,-self.num_ghost-2] + \
        #         45.*w[:,-self.num_ghost-3] - 24.*w[:,-self.num_ghost-4] + 5.*w[:,-self.num_ghost-5]
        # w[:,-self.num_ghost+2] = 35.*w[:,-self.num_ghost-1] - 105.*w[:,-self.num_ghost-2] + \
        #         126.*w[:,-self.num_ghost-3] - 70.*w[:,-self.num_ghost-4] + 15.*w[:,-self.num_ghost-5]

        # for j,evalue in enumerate(evalues):
        #     if evalue <= 0:
        #         w[j,-self.num_ghost:] = 0.

        # Project back to physical space
        # for i in xrange(self.num_ghost):
            # w[:,-self.num_ghost+i] = d0qR + (i+1)*dx*d1qR + ((i+1)*dx)**2/2.*d2qR \
            #     + ((i+1)*dx)**3/6.*d3qR + ((i+1)*dx)**4/24.*d4qR
            
            # w[:,-self.num_ghost+i] = 5.*w[:,i+4] - 10.*w[:,i+3] + 10.*w[:,i+2] - 5.*w[:,i+1] + w[:,i]
            
            # set left-going waves to zero
            # w[0,-self.num_ghost+i] = 0.
            # w[1,-self.num_ghost+i] = 0.

            # w[:,-self.num_ghost+i] = w[:,-self.num_ghost-1]

            
            # array[:,-self.num_ghost+i,...] = np.dot(evr[:,:,-self.num_ghost-1],w[:,-self.num_ghost+i])

            # array[:,-self.num_ghost+i,...] = np.dot(evr[:,:,-1],w[:,-self.num_ghost+i])

            # array[:,-self.num_ghost+i,...] = w[:,-self.num_ghost+i]


    # ========================================================================
    #  Evolution routines
    # ========================================================================
    def accept_reject_step(self,state):
        cfl = self.cfl.get_cached_max()
        if cfl > self.cfl_max:
            return False
        else:
            return True

    def get_dt_new(self):
        cfl = self.cfl.get_cached_max()
        self.dt = min(self.dt_max,self.dt * self.cfl_desired / cfl)

    def get_dt(self,t,tstart,tend,take_one_step):
        cfl = self.cfl.get_cached_max()
        if self.dt_variable and self.dt_old is not None:
            if cfl > 0.0:
                self.get_dt_new()
                self.status['dtmin'] = min(self.dt, self.status['dtmin'])
                self.status['dtmax'] = max(self.dt, self.status['dtmax'])
            else:
                self.dt = self.dt_max
        else:
            self.dt_old = self.dt

        # Adjust dt so that we hit tend exactly if we are near tend
        if not take_one_step:
            if t + self.dt > tend and tstart < tend:
                self.dt = tend - t
            if tend - t - self.dt < 1.e-14*t:
                self.dt = tend - t

    def evolve_to_time(self,solution,tend=None):
        r"""
        Evolve solution from solution.t to tend.  If tend is not specified,
        take a single step.
 
        This method contains the machinery to evolve the solution object in
        ``solution`` to the requested end time tend if given, or one 
        step if not. 

        :Input:
         - *solution* - (:class:`Solution`) Solution to be evolved
         - *tend* - (float) The end time to evolve to, if not provided then
           the method will take a single time step.
 
        :Output:
         - (dict) - Returns the status dictionary of the solver
        """

        if not self._is_set_up:
            self.setup(solution)
 
        if tend == None:
            take_one_step = True
        else:
            take_one_step = False
 
        # Parameters for time-stepping
        tstart = solution.t

        num_steps = 0

        # Setup for the run
        if not self.dt_variable:
            if take_one_step:
                self.max_steps = 1
            else:
                self.max_steps = int((tend - tstart + 1e-10) / self.dt)
                if abs(self.max_steps*self.dt - (tend - tstart)) > 1e-5 * (tend-tstart):
                    raise Exception('dt does not divide (tend-tstart) and dt is fixed!')
        if self.dt_variable == 1 and self.cfl_desired > self.cfl_max:
            raise Exception('Variable time-stepping and desired CFL > maximum CFL')
        if tend <= tstart and not take_one_step:
            self.logger.info("Already at or beyond end time: no evolution required.")
            self.max_steps = 0
 
        # Main time-stepping loop
        for n in xrange(self.max_steps):
 
            state = solution.state
 
            # Keep a backup in case we need to retake a time step
            if self.dt_variable:
                q_backup = state.q.copy('F')
                told = solution.t

            if self.before_step is not None:
                self.before_step(self,solution.states[0])

            # Note that the solver may alter dt during the step() routine
            self.step(solution,take_one_step,tstart,tend)

            # Check to make sure that the Courant number was not too large
            cfl = self.cfl.get_cached_max()
            self.accept_step = self.accept_reject_step(state)
            if self.accept_step:
                # Accept this step
                self.status['cflmax'] = max(cfl, self.status['cflmax'])
                if self.dt_variable==True:
                    solution.t += self.dt 
                else:
                    #Avoid roundoff error if dt_variable=False:
                    solution.t = tstart+(n+1)*self.dt

                # Verbose messaging
                self.logger.debug("Step %i  CFL = %f   dt = %f   t = %f"
                    % (n,cfl,self.dt,solution.t))
 
                self.write_gauge_values(solution)
                # Increment number of time steps completed
                num_steps += 1
                self.status['numsteps'] += 1
 
            else:
                # Reject this step
                self.logger.debug("Rejecting time step, CFL number too large")
                if self.dt_variable:
                    state.q = q_backup
                    solution.t = told
                else:
                    # Give up, we cannot adapt, abort
                    self.status['cflmax'] = \
                        max(cfl, self.status['cflmax'])
                    raise Exception('CFL too large, giving up!')
 
            # See if we are finished yet
            if solution.t >= tend or take_one_step:
                break

        # End of main time-stepping loop -------------------------------------

        if self.dt_variable and solution.t < tend \
                and num_steps == self.max_steps:
            raise Exception("Maximum number of timesteps have been taken")

        return self.status

    def step(self,solution):
        r"""
        Take one step
        
        This method is only a stub and should be overridden by all solvers who
        would like to use the default time-stepping in evolve_to_time.
        """
        raise NotImplementedError("No stepping routine has been defined!")

    # ========================================================================
    #  Gauges
    # ========================================================================
    def write_gauge_values(self,solution):
        r"""Write solution (or derived quantity) values at each gauge coordinate
            to file.
        """
        import numpy as np
        if solution.num_aux == 0:
            aux = None
        for i,gauge in enumerate(solution.state.grid.gauges):
            if self.num_dim == 1:
                ix=gauge[0];
                if solution.num_aux > 0:
                    aux = solution.state.aux[:,ix]
                q=solution.state.q[:,ix]
            elif self.num_dim == 2:
                ix=gauge[0]; iy=gauge[1]
                if solution.num_aux > 0:
                    aux = solution.state.aux[:,ix,iy]
                q=solution.state.q[:,ix,iy]
            p=self.compute_gauge_values(q,aux)
            if not hasattr(p,'__iter__'):
                p = [p]
            t=solution.t
            if solution.state.keep_gauges:
                gauge_data = solution.state.gauge_data
                if len(gauge_data) == len(solution.state.grid.gauges):
                    gauge_data[i]=np.vstack((gauge_data[i],np.append(t,p)))
                else:
                    gauge_data.append(np.append(t,p))
            
            try:
                solution.state.grid.gauge_files[i].write(str(t)+' '+' '.join(str(j) 
                                                         for j in p)+'\n')  
            except IOError:
                raise Exception("Gauge files are not set up correctly. You should call \
                       \nthe method `setup_gauge_files` of the Grid class object \
                       \nbefore any call for `write_gauge_values` from the Solver class.")
                

if __name__ == "__main__":
    import doctest
    doctest.testmod()

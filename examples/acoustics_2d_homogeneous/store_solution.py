#!/usr/bin/env python
# encoding: utf-8

from acoustics_2d import setup
import numpy as np
from clawpack.pyclaw.util import check_diff

def compute_sol(time_integrator,lim_type,use_petsc):
    claw = setup(time_integrator=time_integrator,lim_type=lim_type,use_petsc=use_petsc,solver_type='sharpclaw')
    claw.run()

    state = claw.frames[claw.num_output_times].state
    test_q = state.get_q_global()
    test_pressure = test_q[0,:,:]
    delta = claw.solution.domain.grid.delta

    claw = None

    return test_pressure, delta

def store_sol(time_integrator,lim_type):
    test_pressure,_ = compute_sol(time_integrator,lim_type,use_petsc=0)

    if time_integrator == 'SSP104':
        np.savetxt('verify_sharpclaw.txt',test_pressure)
    elif time_integrator == 'SSPMS32':
        np.savetxt('verify_sharpclaw_lmm.txt',test_pressure)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lim_type", metavar='int', default=1)
    args = parser.parse_args()

    # store data
    store_sol(time_integrator='SSP104',lim_type=args.lim_type)

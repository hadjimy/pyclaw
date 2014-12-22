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

def verify_data(time_integrator,lim_type,use_petsc=1):

    test_pressure, delta = compute_sol(time_integrator,lim_type,use_petsc)
    test_pressure = test_pressure[:].reshape(-1)

    if time_integrator == 'SSP104':
        data_file = 'verify_sharpclaw.txt'
    elif time_integrator == 'SSPMS32':
        data_file = 'verify_sharpclaw_lmm.txt'

    expected_pressure = np.loadtxt(data_file)
    expected_pressure = expected_pressure[:].reshape(-1)

    test_err = np.prod(delta)*np.linalg.norm(expected_pressure-test_pressure)
    print test_err

    print check_diff(expected_pressure, test_pressure, reltol=1e-3)
    print check_diff(0, test_err, abstol=1e-3)


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lim_type", metavar='int', default=1)
    args = parser.parse_args()

    # run again to compare solution with data
    verify_data(time_integrator='SSP104',lim_type=args.lim_type)

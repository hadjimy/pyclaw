#!/bin/bash

echo 'Store solution for SSP104/WENO ...'
python store_solution.py --lim_type=2

echo 'Verify solution for SSP104/WENO in serial ...'
python verify_solution.py --lim_type=2

echo 'Verify solution for SSP104/WENO in parallel ...'
mpirun -n 4 python verify_solution.py --lim_type=2

echo 'Store solution for SSP104/TVD2 ...'
python store_solution.py --lim_type=1

echo 'Verify solution for SSP104/TVD2  in serial ...'
python verify_solution.py --lim_type=1

echo 'Verify solution for SSP104/TVD2 in parallel ...'
mpirun -n 4 python verify_solution.py --lim_type=1


#!/bin/bash
cd /home/claude/drainage
python3 run_experiments.py --split tune --trials 100 --out results/tune.csv
python3 run_experiments.py --split test --trials 100 --out results/test.csv
python3 run_experiments.py --split test --sigma 0.5 --out results/noise05.csv
python3 run_experiments.py --split test --sigma 1.0 --out results/noise10.csv
python3 run_experiments.py --split test --rho 0.5 --out results/corr05.csv
python3 run_experiments.py --split test --rho 1.0 --out results/corr10.csv
python3 run_experiments.py --split test --measure catch --out results/catch.csv
python3 run_experiments.py --split test --measure oracle --out results/oracle.csv
echo ALLDONE

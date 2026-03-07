#!/bin/bash

python3 limit_test.py --c1=0.0 --c2=0.0 --init=random
python3 limit_test.py --c1=0.0 --c2=0.0 --init=identity
python3 limit_test.py --c1=0.0 --c2=0.0 --init=continuum
python3 limit_test.py --c1=-0.1 --c2=0.1 --init=random
python3 limit_test.py --c1=-0.1 --c2=0.1 --init=identity
python3 limit_test.py --c1=-0.1 --c2=0.1 --init=continuum
python3 limit_test.py --c1=-0.5 --c2=0.5 --init=random
python3 limit_test.py --c1=-0.5 --c2=0.5 --init=identity
python3 limit_test.py --c1=-0.5 --c2=0.5 --init=continuum
python3 loss_plots.py


#!/bin/bash

# Run the following python scripts in order
python3 generate_projectile_data.py
python3 _fit.py
python3 _predict.py
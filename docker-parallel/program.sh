#!/bin/bash

export JULIA_NUM_THREADS=$(nproc)
julia /app/main.jl >program.log 

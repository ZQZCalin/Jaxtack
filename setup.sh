#!/bin/bash -l

if [ "$1" = "local" ]; then
    REQ_FILE="deploy/requirements_local.txt"
elif [ "$1" = "" ] || [ "$1" = "scc" ]; then
    module load python3/3.10.12 cuda/12.2
    [ ! -d "env" ] && python -m venv env
    source env/bin/activate
    pip install --upgrade "jax[cuda12]==0.4.31"
    REQ_FILE="deploy/requirements_scc.txt"
else
    echo "Usage: $0 [local|scc]"
    exit 1
fi

pip install -r $REQ_FILE

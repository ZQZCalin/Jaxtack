#!/bin/bash -l

setup_local() {
    module load python3/3.10.12 cuda/12.2
    [ ! -d ".venv" ] && python -m venv .venv
    . .venv/bin/activate
    REQ_FILE="deploy/requirements_local.txt"
    pip install -r "$REQ_FILE"
}

setup_scc() {
    module load python3/3.10.12 cuda/12.2
    [ ! -d ".venv" ] && python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade "jax[cuda12]==0.4.31"
    REQ_FILE="deploy/requirements_scc.txt"
    pip install -r "$REQ_FILE"
}

setup_error() {
    echo "Usage: $0 [local|scc]"
}

if [ "$1" = "local" ]; then
    setup_local
elif [ "$1" = "scc" ]; then
    setup_scc
else
    setup_error
fi


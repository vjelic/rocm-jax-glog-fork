#!/bin/bash

error() {
    echo "$*" >&2
}

die() {
    [ -n "$1" ] && error "$*"
    exit 1
}

python3 build/ci_build \
    test jax-ubu22.rocm700 \
    || die "failure during integration tests"

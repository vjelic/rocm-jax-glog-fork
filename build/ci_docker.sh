#!/bin/bash

error() {
    echo "$*" >&2
}

die() {
    [ -n "$1" ] && error "$*"
    exit 1
}

python3 build/ci_build \
    --rocm-version 7.0.0 \
    --rocm-build-job compute-rocm-dkms-no-npi-hipclang \
    --rocm-build-num 16051 \
    build_dockers \
    -f ubu22 \
    || die "failed to build docker image(s) for testing"

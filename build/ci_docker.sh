#!/bin/bash

error() {
    echo "$*" >&2
}

die() {
    [ -n "$1" ] && error "$*"
    exit 1
}

python3 build/ci_build \
    --rocm-version "$1" \
    --rocm-build-job "$2" \
    --rocm-build-num "$3" \
    build_dockers \
    || die "failed to build docker image(s) for testing"

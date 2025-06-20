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
    --python-versions "$4" \
    --compiler clang dist_wheels \
    || die "jax_rocm_plugin wheel build failed"


# copy wheels from plugin wheel build
mkdir -p wheelhouse
cp jax_rocm_plugin/wheelhouse/* wheelhouse/

for whl in wheelhouse/*; do
    curl -u "$ARTIFACTORY_USER:$ARTIFACTORY_PASSWORD" \
        -T "$whl" \
        "https://compute-artifactory.amd.com/artifactory/generic-local/rocm-jax/$(basename "$whl")" \
        || die "Failed to upload $(basename "$whl") to Artifactory"
done

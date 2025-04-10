

# gfx targets for which XLA and jax custom call kernels are built for
AMDGPU_TARGETS ?= "gfx906,gfx908,gfx90a,gfx942,gfx1030,gfx1100,gfx1101,gfx1200,gfx1201"

# customize to a single arch for local dev builds to reduce compile time
#AMDGPU_TARGETS ?= "gfx908"

.PHONY: test clean install

.default: dist


clean:
	rm -rf dist

dist:
	python3 ./build/build.py build \
            --use_clang=true \
            --wheels=jax-rocm-plugin,jax-rocm-pjrt \
            --rocm_path=/opt/rocm/ \
            --rocm_version=60 \
            --rocm_amdgpu_targets=${AMDGPU_TARGETS} \
	    --bazel_options="--override_repository=xla=../xla" \
            --verbose \
            --clang_path=/usr/lib/llvm-18/bin/clang

install: dist 
	pip install --force-reinstall dist/*

test: install
	python3 -m unittest discover -v tests

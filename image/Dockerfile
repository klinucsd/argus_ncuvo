# =============================================================================
# ARGUS-NCUVO: ARGUS + CDeep3M-anywhere for NRP JupyterHub
# =============================================================================
# An ARGUS release image with Willy's CDeep3M-anywhere (EM image segmentation:
# ncnn prediction on CPU or Vulkan GPU, PyTorch training) added on top, as
# ARGUS-FEDER adds TokSearch, plus the cdeep3m skill. Built from the REPOSITORY ROOT:
#
#   docker build -f image/Dockerfile -t kaiucsd/argus-ncuvo:<version>-cpu .
#   docker build -f image/Dockerfile -t kaiucsd/argus-ncuvo:<version>-cu118 \
#       --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu118 --build-arg TORCH_SPEC="torch==2.7.1" .
#
# The CDeep3M steps follow CDeep3m-anywhere/docker/Dockerfile; they differ only in the
# runtime base (ARGUS instead of plain Ubuntu) and the Python environment (the base
# image's conda instead of a venv). No model weights or data are in the image.
#
# ARGUS_IMAGE can also be quay.io/jupyter/base-notebook:latest, for CDeep3M without ARGUS.
# =============================================================================

ARG ARGUS_IMAGE=kaiucsd/argus-dev:v1.5.11-rc3

# ---------- stage 1: ncnn tools, CDeep3M2 and CDeep3M-anywhere sources, caffe.proto ----------
FROM ubuntu:24.04 AS builder
ARG NCNN_TAG=20261020
ARG CDEEP3M_ANYWHERE_COMMIT=475277e
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build git ca-certificates libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*
# Only the conversion tools (caffe2ncnn, ncnn2table, ncnn2int8); prediction uses the ncnn Python wheel.
RUN git clone --depth 1 --branch ${NCNN_TAG} https://github.com/Tencent/ncnn.git /src/ncnn \
    && cmake -S /src/ncnn -B /src/ncnn/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=OFF -DNCNN_BUILD_TOOLS=ON -DNCNN_SIMPLEOCV=ON \
        -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_BUILD_BENCHMARK=OFF -DNCNN_PYTHON=OFF \
    && cmake --build /src/ncnn/build -j ${JOBS} \
    && mkdir -p /opt/ncnn_build/tools/caffe /opt/ncnn_build/tools/quantize \
    && cp /src/ncnn/build/tools/caffe/caffe2ncnn /opt/ncnn_build/tools/caffe/ \
    && cp /src/ncnn/build/tools/quantize/ncnn2table /src/ncnn/build/tools/quantize/ncnn2int8 /opt/ncnn_build/tools/quantize/
RUN git clone https://github.com/CRBS/cdeep3m2.git /opt/cdeep3m2 \
    && git -C /opt/cdeep3m2 checkout 67d6308 && rm -rf /opt/cdeep3m2/.git
RUN git clone https://github.com/CRBS/CDeep3m-anywhere.git /opt/cdeep3m-anywhere \
    && git -C /opt/cdeep3m-anywhere checkout ${CDEEP3M_ANYWHERE_COMMIT} && rm -rf /opt/cdeep3m-anywhere/.git
RUN git clone --filter=blob:none --no-checkout https://github.com/coleslaw481/caffe_nd_sense_segmentation.git /src/fork \
    && git -C /src/fork show 6566b5d:src/caffe/proto/caffe.proto > /opt/caffe.proto

# ---------- stage 2: runtime on the ARGUS image (Ubuntu 24.04, jovyan, NRP JupyterHub) ----------
FROM ${ARGUS_IMAGE}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ARG TORCH_SPEC=torch

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git patch parallel bc time util-linux procps curl protobuf-compiler \
        libgomp1 libprotobuf32t64 libvulkan1 vulkan-tools libegl1 libxext6 libx11-6 \
    && rm -rf /var/lib/apt/lists/*
# Vulkan on NVIDIA (from Willy's Dockerfile): ICD files older container toolkits do not mount.
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.2.175"}}\n' \
        > /usr/share/vulkan/icd.d/nvidia_icd.json \
    && printf '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}\n' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Python packages go into the base image's conda environment, as jovyan (files stay
# jovyan-owned, no recursive chmod layer). Checked: this only adds packages, it changes
# none of ARGUS's. JupyterLab stays pinned to NRP JupyterHub's runtime.
USER ${NB_UID}
RUN pip install --no-cache-dir "jupyterlab==4.2.4" "notebook==7.2.2" \
        ncnn numpy h5py pillow opencv-python-headless scikit-image joblib imageio \
        matplotlib pandas pyyaml requests protobuf \
    && pip install --no-cache-dir --index-url ${TORCH_INDEX} ${TORCH_SPEC}

USER root
COPY --from=builder /opt/ncnn_build /opt/ncnn_build
COPY --from=builder /opt/cdeep3m2 /opt/cdeep3m2
COPY --from=builder /opt/cdeep3m-anywhere /opt/cdeep3m-anywhere
COPY --from=builder /opt/caffe.proto /opt/caffe.proto
COPY image/pod_test.sh /opt/argus-ncuvo/pod_test.sh
# The skill, where ARGUS looks for skills (as its core skills are installed)
COPY --chown=${NB_UID}:${NB_GID} skills/cdeep3m /home/${NB_USER}/.deepagents/agent/skills/cdeep3m
# As in Willy's Dockerfile: patch CDeep3M2, make its scripts executable and run them with the
# environment's python. chmod only these folders: /opt also holds conda.
RUN cd /opt/cdeep3m2 \
    && patch -p1 < /opt/cdeep3m-anywhere/patches/cdeep3m2-67d6308-device-backend.patch \
    && cp /opt/cdeep3m-anywhere/ncnn/ncnn_predict.py . \
    && sed -i '1s|^#!.*python.*$|#!/usr/bin/env python3|' $(find . -name '*.py') \
    && chmod +x *.sh *.py /opt/cdeep3m-anywhere/ncnn/*.sh /opt/cdeep3m-anywhere/tools/*.sh \
        /opt/cdeep3m-anywhere/singularity/gpu_check.sh /opt/argus-ncuvo/pod_test.sh \
    && chmod -R a+rX /opt/cdeep3m2 /opt/cdeep3m-anywhere /opt/argus-ncuvo /opt/ncnn_build /opt/caffe.proto

ENV CDEEP3M2=/opt/cdeep3m2 \
    NCNN_BUILD=/opt/ncnn_build \
    CAFFE_PROTO=/opt/caffe.proto \
    PYTHONPATH=/opt/cdeep3m-anywhere:/home/jovyan/.deepagents/agent/skills/cdeep3m \
    PATH=/opt/argus-ncuvo:/opt/cdeep3m2:/opt/cdeep3m-anywhere/ncnn:/opt/cdeep3m-anywhere/tools:/opt/cdeep3m-anywhere/singularity:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all

USER ${NB_UID}

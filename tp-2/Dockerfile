FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv curl build-essential git \
    && rm -rf /var/lib/apt-get/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

RUN pip3 install --no-cache-dir uv py-spy

ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV PATH="/root/.cargo/bin:${CUDA_PATH}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${CUDA_PATH}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
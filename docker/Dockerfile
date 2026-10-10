# SAVA development environment: toolchain for the simulator (C++/CMake) and
# the compiler stack (IREE/MLIR Python packages).
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# Max-FutureTime tolerates the clock drift common in WSL2/Docker Desktop VMs.
RUN apt-get -o Acquire::Max-FutureTime=86400 update && apt-get install -y --no-install-recommends \
        build-essential clang lld cmake ninja-build git ca-certificates \
        python3 python3-venv python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /workspace

# The in-tree Python package (assembler, runtime, compiler).
ENV PYTHONPATH=/workspace/python

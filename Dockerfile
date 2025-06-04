# CUDA 12.3 on Ubuntu 22.04 for RevDecode

FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

ARG GHIDRA_VER=11.0
ARG GHIDRA_DATE=20231222

ENV GHIDRA_HOME=/opt/ghidra \
    DEBIAN_FRONTEND=noninteractive \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
    

# Toolchain
# adding essentials for building RevDecode linux version + Ghidra + vim

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates \
        python3 python3-pip \
        nlohmann-json3-dev \
        openjdk-17-jdk \
        unzip \
        vim && \
    rm -rf /var/lib/apt/lists/* && \
    \
    # installing Ghidra 11
    curl -fsSL -o /tmp/ghidra.zip \
      "https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_${GHIDRA_VER}_build/ghidra_${GHIDRA_VER}_PUBLIC_${GHIDRA_DATE}.zip" && \
    unzip -q /tmp/ghidra.zip -d /opt && \
    mv /opt/ghidra_${GHIDRA_VER}_PUBLIC "${GHIDRA_HOME}" && \
    rm /tmp/ghidra.zip && \
    ln -s "${GHIDRA_HOME}/ghidraRun" /usr/local/bin/ghidra

# installing numpy for revdecode script(s)
RUN pip install numpy

WORKDIR /workspace
COPY ./src /workspace
CMD ["/bin/bash"]

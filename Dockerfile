ARG NVIDIA_VERSION=24.06


FROM nvcr.io/nvidia/pytorch:$NVIDIA_VERSION-py3 AS base

ENV HUGGINGFACE_HUB_CACHE="/mnt/hub_cache/hf"
ENV TORCH_HOME="/mnt/hub_cache/torch"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workdir
COPY src src


FROM base AS unit-test

RUN pip install pytest

COPY tests tests
COPY pyproject.toml pyproject.toml

ENTRYPOINT [ "pytest" ]


FROM base AS final

WORKDIR /workdir/src

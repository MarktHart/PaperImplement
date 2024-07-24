#!/bin/bash

set -e

OLD_NVIDIA_VERSION=$(cat helpers/version/verified_nvidia_version.txt)
LATEST_NVIDIA_VERSION=$(date -d "$(date +%Y-%m-1) -1 month" +%y.%m)

IMAGE_NAME="paper-implement-test"

build_and_test () {
    docker build --build-arg="NVIDIA_VERSION=$1" -t $IMAGE_NAME:$1 --no-cache --target=unit-test .
    docker run --rm -v ~/.cache/huggingface/hub:/mnt/hub_cache/hf -v ~/.cache/torch/hub:/mnt/hub_cache/torch --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $IMAGE_NAME:$1
}

if [ $OLD_NVIDIA_VERSION != $LATEST_NVIDIA_VERSION ]; then
    build_and_test $OLD_NVIDIA_VERSION
fi

build_and_test $LATEST_NVIDIA_VERSION

echo $LATEST_NVIDIA_VERSION > helpers/version/verified_nvidia_version.txt
docker run --rm --entrypoint pip $IMAGE_NAME:$LATEST_NVIDIA_VERSION freeze > helpers/version/verified_pip_freeze.txt

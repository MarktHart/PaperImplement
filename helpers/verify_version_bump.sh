#!/bin/bash

set -e

OLD_NVIDIA_VERSION=$(cat helpers/verified_nvidia_version.txt)
LATEST_NVIDIA_VERSION=$(date -d "$(date +%Y-%m-1) -1 month" +%y.%m)

build_and_test () {
    docker build --build-arg="NVIDIA_VERSION=$1" -t $2 --no-cache --target=test .
    docker run --rm -v ~/.cache/huggingface/hub:/mnt/hub_cache/hf -v ~/.cache/torch/hub:/mnt/hub_cache/torch --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $2
}

if [ $OLD_NVIDIA_VERSION != $LATEST_NVIDIA_VERSION ]; then
    build_and_test $OLD_NVIDIA_VERSION implement-old-test
fi

build_and_test $LATEST_NVIDIA_VERSION implement-latest-test

echo $LATEST_NVIDIA_VERSION > helpers/verified_nvidia_version.txt

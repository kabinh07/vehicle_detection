#!/bin/bash
set -e
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=best.torchscript
NAME=vehicle
VERSION=$1
EXTRA=$2
if [ $EXTRA ]; then
    EXTRA="--extra-files ${EXTRA}"
    else EXTRA=""
fi
if [ -z $VERSION ];then
  VERSION='1.0'
fi
echo "VERSION: ${VERSION}"
# create mar
docker run --rm \
-v $PWD/:/home/model-server \
-v $PWD/model-store:/model-store \
-v $PWD/models:/models \
--entrypoint /bin/bash \
--workdir /home/model-server \
$CONTAINER \
-c \
"torch-model-archiver \
--model-name ${NAME} \
--version ${VERSION} \
--serialized-file /models/${MODEL} \
--handler handler.py \
--requirements-file requirements.txt \
${EXTRA} \
--force \
&& mv ${NAME}.mar /model-store/
"

docker run --rm -itd --name vehicle \
-p 127.0.0.1:8080:8080 \
-p 127.0.0.1:8081:8081 \
-p 127.0.0.1:8082:8082 \
-v $(pwd)/model-store:/home/model-server/model-store \
$CONTAINER \
torchserve --start --ncs \
--model-store model-store \
--models vehicle=model-store/vehicle.mar \
--disable-token-auth
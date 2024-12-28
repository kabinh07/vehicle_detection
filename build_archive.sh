#!/bin/bash
set -e
CONTAINER=pytorch/torchserve:latest-cpu
MODEL=best.pt
NAME=vehicle-detection
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
-v $PWD/model_store:/model_store \
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
&& mv ${NAME}.mar /model_store/
"
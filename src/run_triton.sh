#!/bin/bash

MODEL_REPOSITORY=s3://nvidia-triton/model-repository

## GET MODELS FROM S3 INTO THE /TMP FOLDER
aws s3 sync ${MODEL_REPOSITORY} /tmp/model_repository

# RUN TRITON
/opt/tritonserver/bin/tritonserver --model-repository=/tmp/model_repository

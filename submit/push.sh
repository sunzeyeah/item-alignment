#!/usr/bin/env bash

#Compress-Archive -Update -Path .\similarity.py,.\deepAI_result.jsonl -DestinationPath result.zip

TAG="ensemble-threshold_0.0"

docker build -t registry.cn-shanghai.aliyuncs.com/ccks2022_task9_subtask2/submit:$TAG .
docker push registry.cn-shanghai.aliyuncs.com/ccks2022_task9_subtask2/submit:$TAG

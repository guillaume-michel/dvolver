#!/bin/bash

kubectl create -f rabbitmq-service.yaml
kubectl create -f rabbitmq-management-service.yaml

kubectl create -f rabbitmq-deployment.yaml

kubectl create -f dvolver-server-deployment.yaml

kubectl create -f dvolver-worker-gpu-deployment.yaml

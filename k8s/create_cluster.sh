#!/bin/bash

# https://cloud.google.com/kubernetes-engine/docs/how-to/creating-a-cluster

PROJECT="<PROJECT_NAME>"
CLUSTER_NAME="dvolver-k8s"
CLUSTER_VERSION="1.10.7-gke.11"
ZONE="<ZONE>"

DEFAULT_NUM_NODES="1"
DEFAULT_MIN_NODES="0"
DEFAULT_MAX_NODES="7"

DVOLVER_BROKER_NUM_NODES="0"
DVOLVER_BROKER_MIN_NODES="0"
DVOLVER_BROKER_MAX_NODES="2"

DVOLVER_SERVER_NUM_NODES="0"
DVOLVER_SERVER_MIN_NODES="0"
DVOLVER_SERVER_MAX_NODES="2"

DVOLVER_WORKER_GPU_NUM_NODES="0"
DVOLVER_WORKER_GPU_MIN_NODES="0"
DVOLVER_WORKER_GPU_MAX_NODES="50"

DVOLVER_WORKER_TPU_NUM_NODES="0"
DVOLVER_WORKER_TPU_MIN_NODES="0"
DVOLVER_WORKER_TPU_MAX_NODES="150"


# create cluster with default-pool
gcloud beta container --project $PROJECT clusters create $CLUSTER_NAME --zone $ZONE --username "admin" --cluster-version $CLUSTER_VERSION --addons HorizontalPodAutoscaling,HttpLoadBalancing --enable-tpu --issue-client-certificate --enable-cloud-logging --enable-cloud-monitoring --enable-ip-alias --network "projects/$PROJECT/global/networks/default" --subnetwork "projects/$PROJECT/regions/europe-west4/subnetworks/default" --machine-type "n1-standard-1" --disk-size "20" --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --image-type "COS" --disk-type "pd-standard" --enable-autoscaling --no-enable-autoupgrade --enable-autorepair --num-nodes $DEFAULT_NUM_NODES --min-nodes $DEFAULT_MIN_NODES --max-nodes $DEFAULT_MAX_NODES

# create dvolver-broker-pool
gcloud beta container --project $PROJECT node-pools create "dvolver-broker-pool" --cluster $CLUSTER_NAME --zone $ZONE --node-version $CLUSTER_VERSION --machine-type "n1-standard-1" --disk-size "20" --node-labels dvolver=broker --scopes "https://www.googleapis.com/auth/devstorage.read_only","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --image-type "COS" --disk-type "pd-standard" --enable-autoscaling --no-enable-autoupgrade --enable-autorepair --num-nodes $DVOLVER_BROKER_NUM_NODES --min-nodes $DVOLVER_BROKER_MIN_NODES --max-nodes $DVOLVER_BROKER_MAX_NODES --node-taints app=dvolver:NoSchedule

# create dvolver-server-pool
gcloud beta container --project $PROJECT node-pools create "dvolver-server-pool" --cluster $CLUSTER_NAME --zone $ZONE --node-version $CLUSTER_VERSION --machine-type "n1-standard-1" --disk-size "30" --node-labels dvolver=server --scopes "https://www.googleapis.com/auth/cloud-platform" --image-type "COS" --disk-type "pd-standard" --enable-autoscaling --no-enable-autoupgrade --enable-autorepair --num-nodes $DVOLVER_SERVER_NUM_NODES --min-nodes $DVOLVER_SERVER_MIN_NODES --max-nodes $DVOLVER_SERVER_MAX_NODES --node-taints app=dvolver:NoSchedule

# create dvolver-worker-gpu-pool
gcloud beta container --project $PROJECT node-pools create "dvolver-worker-gpu-pool" --cluster $CLUSTER_NAME --zone $ZONE --node-version $CLUSTER_VERSION --machine-type "n1-standard-4" --accelerator "type=nvidia-tesla-v100,count=1" --disk-size "30" --node-labels dvolver=worker-gpu --scopes "https://www.googleapis.com/auth/cloud-platform" --image-type "COS" --disk-type "pd-standard" --enable-autoscaling --no-enable-autoupgrade --enable-autorepair --preemptible --num-nodes $DVOLVER_WORKER_GPU_NUM_NODES --min-nodes $DVOLVER_WORKER_GPU_MIN_NODES --max-nodes $DVOLVER_WORKER_GPU_MAX_NODES --node-taints app=dvolver:NoSchedule

# setup nvidia driver auto install on gpu nodes
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/stable/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# create dvolver-worker-tpu-pool
gcloud beta container --project $PROJECT node-pools create "dvolver-worker-tpu-pool" --cluster $CLUSTER_NAME --zone $ZONE --node-version $CLUSTER_VERSION --machine-type "n1-standard-2" --disk-size "30" --node-labels dvolver=worker-tpu --scopes "https://www.googleapis.com/auth/cloud-platform" --image-type "COS" --disk-type "pd-standard" --enable-autoscaling --no-enable-autoupgrade --enable-autorepair --num-nodes $DVOLVER_WORKER_TPU_NUM_NODES --min-nodes $DVOLVER_WORKER_TPU_MIN_NODES --max-nodes $DVOLVER_WORKER_TPU_MAX_NODES --node-taints app=dvolver:NoSchedule

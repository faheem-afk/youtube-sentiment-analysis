#!/bin/bash

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 596514762357.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker Image"
docker pull 596514762357.dkr.ecr.us-east-1.amazonaws.com/ysa-ecr:latest

echo "Checking for existing container"
if [ "$(docker ps -q -f name=container-app)" ]; then
    echo "Stopping existing container..."
    docker stop container-app
    echo "Removing existing container..."
    docker rm container-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name container-app 596514762357.dkr.ecr.us-east-1.amazonaws.com/ysa-ecr:latest
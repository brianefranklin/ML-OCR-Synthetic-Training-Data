#!/bin/sh
echo "stopping any existing containers for vscode-remote-dev"
docker stop $(docker ps -q --filter "name=vscode-dev-container")
echo "deleting existing dev container"
docker rm vscode-dev-container
echo "building new container"
docker build -t vscode-remote-dev .
echo "clearing old ssh known host"
ssh-keygen -R "[localhost]:2222"
echo "running container"
docker run -d --name vscode-dev-container -p 2222:22 -v "$(pwd)":/home/vscode/workspace vscode-remote-dev

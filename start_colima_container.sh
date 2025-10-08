#!/bin/bash

# A script to ensure Colima is running, then switch Docker contexts to
# start a specific development container, and finally switch back.
#
# FLAGS:
# --rebuild      : Stops/removes the container, builds a new image, and clears SSH keys.
# --reconfigure  : Stops/deletes the Colima VM to apply new CPU/memory settings.

# --- CONFIGURATION ---
# Set the Colima profile name you want to check/start. 'default' is the standard.
PROFILE_NAME="default"

# --- Set Colima resource allocation ---
# Default: 4 CPUs, 8 GiB Memory
COLIMA_CPU=4
COLIMA_MEMORY=8

# Set the name of your development container
DEV_CONTAINER_NAME="vscode-dev-container"

# Set the tag for your Docker image
DEV_IMAGE_TAG="my-dev-image"

# --- Add any additional flags for the 'docker run' command here ---
# Note: Use single quotes if you need values like $(pwd) to be evaluated when the container is run.
ADDITIONAL_DOCKER_FLAGS='-v "$(pwd)":/home/vscode/workspace -v /Users/brianfranklin/Documents/GitHub/gemini_persistent_history:/home/vscode/.gemini'

# This command constructs the final docker run command from the variables above.
DEV_CONTAINER_COMMAND="docker run -d --name $DEV_CONTAINER_NAME -p 2222:22 $ADDITIONAL_DOCKER_FLAGS $DEV_IMAGE_TAG"
# --- END CONFIGURATION ---


# --- FLAG PARSING ---
# Loop through arguments to check for flags
REBUILD_FLAG=false
RECONFIGURE_FLAG=false
for arg in "$@"; do
  case $arg in
    --rebuild)
      REBUILD_FLAG=true
      shift
      ;;
    --reconfigure)
      RECONFIGURE_FLAG=true
      shift
      ;;
  esac
done
# --- END FLAG PARSING ---


# --- FUNCTIONS ---
rebuild_container() {
    echo "Rebuilding container..."
    if [ "$(docker ps -a -q -f name=^/${DEV_CONTAINER_NAME}$)" ]; then
        echo "--> Stopping and removing existing container..."
        docker stop $DEV_CONTAINER_NAME
        docker rm $DEV_CONTAINER_NAME
    fi
    echo "--> Building new container image with tag '$DEV_IMAGE_TAG'..."
    docker build -t $DEV_IMAGE_TAG .
    echo "--> Clearing old SSH known host key for [localhost]:2222..."
    ssh-keygen -R "[localhost]:2222"
    echo "Rebuild complete."
}

reconfigure_colima() {
    echo "Reconfiguring Colima VM to apply new resource settings..."
    echo "--> Stopping Colima profile '$PROFILE_NAME'..."
    colima stop "$PROFILE_NAME"
    echo "--> Deleting existing Colima VM for profile '$PROFILE_NAME'..."
    colima delete "$PROFILE_NAME"
    echo "Colima VM has been deleted. It will be recreated with new settings."
}
# --- END FUNCTIONS ---


# 0. Handle VM reconfiguration if flagged
if [ "$RECONFIGURE_FLAG" = true ]; then
    reconfigure_colima
fi


# 1. Ensure Colima is running
echo "Checking status of Colima profile: '$PROFILE_NAME'..."
# --- MODIFIED LINE ---
# This check is more robust and works with newer Colima versions.
if colima status --profile "$PROFILE_NAME" | grep -q "colima is running"; then
    echo "✅ Colima is already running."
else
    echo "🟡 Colima is not running. Starting it now with $COLIMA_CPU CPUs and ${COLIMA_MEMORY}GB Memory..."
    colima start "$PROFILE_NAME" --cpu "$COLIMA_CPU" --memory "$COLIMA_MEMORY"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start Colima. Aborting."
        exit 1
    fi
fi

echo "--------------------------------------------------"

# 2. Manage context, rebuild (if requested), and start the container
echo "Preparing to start dev container..."
CURRENT_CONTEXT=$(docker context show)
echo "Current Docker context is: '$CURRENT_CONTEXT'"

if [ "$CURRENT_CONTEXT" != "colima" ]; then
    echo "Switching context to 'colima'..."
    docker context use colima
fi

if [ "$REBUILD_FLAG" = true ]; then
    rebuild_container
fi

if [ "$(docker ps -a -q -f name=^/${DEV_CONTAINER_NAME}$)" ]; then
    echo "🟡 Dev container '$DEV_CONTAINER_NAME' already exists. Skipping start."
else
    echo "🚀 Starting dev container '$DEV_CONTAINER_NAME'..."
    eval $DEV_CONTAINER_COMMAND
fi

if [ "$CURRENT_CONTEXT" != "colima" ]; then
    echo "Switching context back to '$CURRENT_CONTEXT'..."
    docker context use "$CURRENT_CONTEXT"
else
    echo "Remaining in 'colima' context."
fi

echo "✅ Script finished."



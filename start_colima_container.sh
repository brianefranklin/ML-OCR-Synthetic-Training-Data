#!/bin/bash

# A script to ensure Colima is running, then switch Docker contexts to
# start a specific development container, and finally switch back.
# Can optionally rebuild the container image if run with --rebuild.

# --- CONFIGURATION ---
# Set the Colima profile name you want to check/start. 'default' is the standard.
PROFILE_NAME="default"

# Set the name of your development container
DEV_CONTAINER_NAME="vscode-dev-container"

# Set the tag for your Docker image
DEV_IMAGE_TAG="vscode-remote-dev"

# IMPORTANT: Customize this command to start your specific container
# Ensure you use the $DEV_CONTAINER_NAME and $DEV_IMAGE_TAG variables.
#DEV_CONTAINER_COMMAND="docker run -d --name $DEV_CONTAINER_NAME -p 2222:22 $DEV_IMAGE_TAG"
DEV_CONTAINER_COMMAND="docker run -d --name $DEV_CONTAINER_NAME -p 2222:22 -v "$(pwd)":/home/vscode/workspace $DEV_IMAGE_TAG"
# --- END CONFIGURATION ---


# --- REBUILD FUNCTION ---
# This function handles stopping, removing, building, and clearing SSH keys.
rebuild_container() {
    echo "Rebuilding container..."

    # Stop and remove the existing container if it exists
    if [ "$(docker ps -a -q -f name=^/${DEV_CONTAINER_NAME}$)" ]; then
        echo "--> Stopping and removing existing container..."
        docker stop $DEV_CONTAINER_NAME
        docker rm $DEV_CONTAINER_NAME
    fi

    # Build the new image
    echo "--> Building new container image with tag '$DEV_IMAGE_TAG'..."
    docker build -t $DEV_IMAGE_TAG .

    # Clear the old SSH host key
    echo "--> Clearing old SSH known host key for [localhost]:2222..."
    ssh-keygen -R "[localhost]:2222"
    
    echo "Rebuild complete."
}
# --- END REBUILD FUNCTION ---


# 1. Ensure Colima is running
echo "Checking status of Colima profile: '$PROFILE_NAME'..."
if colima status --profile "$PROFILE_NAME" | grep -q "Running: true"; then
    echo "✅ Colima is already running."
else
    echo "🟡 Colima is not running. Starting it now..."
    colima start "$PROFILE_NAME"
    # Check if the start command was successful
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start Colima. Aborting."
        exit 1
    fi
fi

echo "--------------------------------------------------"

# 2. Manage context, rebuild (if requested), and start the container
echo "Preparing to start dev container..."

# Get the current docker context and save it
CURRENT_CONTEXT=$(docker context show)
echo "Current Docker context is: '$CURRENT_CONTEXT'"

# Switch to the colima context if we are not already there
if [ "$CURRENT_CONTEXT" != "colima" ]; then
    echo "Switching context to 'colima'..."
    docker context use colima
fi

# Check for a --rebuild flag as the first argument
if [ "$1" == "--rebuild" ]; then
    rebuild_container
fi

# Check if the dev container is already running or exists
if [ "$(docker ps -a -q -f name=^/${DEV_CONTAINER_NAME}$)" ]; then
    echo "🟡 Dev container '$DEV_CONTAINER_NAME' already exists. Skipping start."
else
    echo "🚀 Starting dev container '$DEV_CONTAINER_NAME'..."
    # The 'eval' command ensures that the command string with variables and quotes is executed correctly.
    eval $DEV_CONTAINER_COMMAND
fi

# Switch back to the original context if we changed it
if [ "$CURRENT_CONTEXT" != "colima" ]; then
    echo "Switching context back to '$CURRENT_CONTEXT'..."
    docker context use "$CURRENT_CONTEXT"
else
    echo "Remaining in 'colima' context."
fi

echo "✅ Script finished."



# Use the latest Long-Term Support (LTS) version of Ubuntu
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# 1. APPLY SECURITY UPDATES & INSTALL PACKAGES
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    openssh-server \
    sudo \
    wget \
    ca-certificates \
    git \
    curl \
    gnupg \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libraqm-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    python3-psutil && \
    # Clean up the package cache
    rm -rf /var/lib/apt/lists/*

# --- START: MODIFIED NODE.JS & NPM INSTALLATION ---
# This is the new, recommended way to install Node.js and npm

# Download and run the NodeSource setup script for Node.js version 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -

# Install Node.js from the newly added repository. This will also install the correct npm.
RUN apt-get install -y nodejs
# --- END: MODIFIED NODE.JS & NPM INSTALLATION ---


# Get gemini-cli up and running
RUN npm install -g @google/gemini-cli --no-interactive --quiet

# Get claude-cli up and running
RUN npm install -g @anthropic-ai/claude-code --no-interactive --quiet

# --- START: PYTHON DEPENDENCY INSTALLATION ---
# Copy the requirements file
COPY requirements.txt .

# Install Python packages
RUN python3 -m pip install --no-cache-dir -r requirements.txt
# --- END: PYTHON DEPENDENCY INSTALLATION ---


# 2. CREATE A NON-ROOT USER
RUN useradd -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# 3. SET UP SSH FOR PASSWORDLESS AUTHENTICATION
USER vscode
WORKDIR /home/vscode

# --- START: PERSIST GEMINI HISTORY ---
# Remove the default directory and link it to the persistent workspace location
RUN rm -rf /home/vscode/.gemini && \
    ln -s /home/vscode/workspace/gemini_persistent_history/.gemini/ /home/vscode/
# --- END: PERSIST GEMINI HISTORY ---

# --- START: PERSIST CLAUDE HISTORY ---
# Remove the default directory and link it to the persistent workspace location
RUN rm -rf /home/vscode/.claude && \
    ln -s /home/vscode/workspace/claude_persistent_history/.claude/ /home/vscode/
# --- END: PERSIST CLAUDE HISTORY ---
    

RUN mkdir -p /home/vscode/.ssh
COPY --chown=vscode:vscode id_rsa.pub /home/vscode/.ssh/authorized_keys
RUN chmod 700 /home/vscode/.ssh && \
    chmod 600 /home/vscode/.ssh/authorized_keys

# Switch back to the root user for final system configurations
USER root

# 4. CONFIGURE & HARDEN THE SSH SERVER
RUN mkdir /var/run/sshd
RUN ssh-keygen -A
RUN echo "PermitRootLogin no" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

# Expose port 22 to the host machine.
EXPOSE 22

# 5. START THE SSH SERVER
CMD ["/usr/sbin/sshd", "-D"]

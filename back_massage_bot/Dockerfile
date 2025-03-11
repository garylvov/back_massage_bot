FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    build-essential \
    sudo \
    git \
    tree

# Define build arguments for user configuration
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=1000

# Create user and configure sudo without password
RUN groupadd -g ${USER_GID} ${USERNAME} && \
    useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

# Switch to user for subsequent operations
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    echo 'export PATH="/home/${USERNAME}/.pixi/bin:$PATH"' >> ~/.bashrc

# Set PATH environment variable to include Pixi
ENV PATH="/home/${USERNAME}/.pixi/bin:${PATH}"

# Create and set working directory with correct permissions
WORKDIR /back_massage_bot
USER root
RUN mkdir -p /models/dense_pose && \
    chown -R ${USERNAME}:${USERNAME} /models
RUN mkdir -p /back_massage_bot && \
    chown -R ${USERNAME}:${USERNAME} /back_massage_bot
USER ${USERNAME}

# Copy pixi configuration
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/pixi.toml /back_massage_bot/pixi.toml
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/pyproject.toml /back_massage_bot/pyproject.toml
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/requirements.txt /back_massage_bot/requirements.txt

# Create pixi environment
RUN cd /back_massage_bot && pixi install

# Install DensePose after PyTorch is installed
RUN cd /back_massage_bot && pixi run install-densepose

# Copy post-entry hooks
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/post-entry-hooks.sh /back_massage_bot/post-entry-hooks.sh

# Download DensePose models using pixi tasks
RUN cd /back_massage_bot && pixi run get-all-models

# Run post-entry hooks to set up environment
RUN bash /back_massage_bot/post-entry-hooks.sh

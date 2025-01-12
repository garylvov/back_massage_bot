FROM nvidia/cuda:12.6.3-base-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

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

# Install Miniforge in user's home directory
ENV CONDA_DIR=/home/${USERNAME}/miniforge3
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p ${CONDA_DIR} && \
    rm Miniforge3-$(uname)-$(uname -m).sh

# Set environment path
ENV PATH="${CONDA_DIR}/bin:$PATH"

# Initialize conda for bash and disable auto activation of base
RUN conda init bash && \
    echo "conda config --set auto_activate_base false" >> ~/.bashrc

# Create and set working directory with correct permissions
WORKDIR /therapeutech
USER root
RUN mkdir -p /therapeutech && \
    chown -R ${USERNAME}:${USERNAME} /therapeutech
USER ${USERNAME}

# Create conda environment
COPY --chown=${USERNAME}:${USERNAME} therapeutech/environment.yml /therapeutech/environment.yml
RUN conda env create -f /therapeutech/environment.yml && \
    conda clean -a -y && \
    conda init bash

# Copy the rest of the project files
COPY --chown=${USERNAME}:${USERNAME} therapeutech /therapeutech

# Install package
RUN conda run -n therapeutech pip install -e /therapeutech

# Ensure conda environment is activated for interactive sessions
RUN echo "conda activate therapeutech" >> ~/.bashrc

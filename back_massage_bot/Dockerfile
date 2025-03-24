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
ARG USERNAME=developer
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
WORKDIR /back_massage_bot
USER root
RUN mkdir -p /models/  && \
    chmod -R 777 /models
RUN mkdir -p /back_massage_bot && \
    chmod -R 777 /back_massage_bot
USER ${USERNAME}

RUN wget -O /models/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
RUN wget -O /models/yolo11m.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
# Create conda environment
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/environment.yml /back_massage_bot/environment.yml
RUN conda env create -f /back_massage_bot/environment.yml && \
    conda clean -a -y && \
    conda init bash


SHELL ["conda", "run", "-n", "back_massage_bot", "/bin/bash", "-c"]
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/post-entry-hooks.sh /back_massage_bot/post-entry-hooks.sh
RUN

RUN bash post-entry-hooks.sh

# Copy the rest of the project files with correct permissions
USER root
COPY --chown=${USERNAME}:${USERNAME} --chmod=777 back_massage_bot /back_massage_bot
USER ${USERNAME}

# Install package
RUN pip install -r /back_massage_bot/requirements.txt && pip install -e /back_massage_bot

# Ensure conda environment is activated for interactive sessions
RUN echo "conda activate back_massage_bot" >> ~/.bashrc

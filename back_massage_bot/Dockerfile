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
RUN mkdir -p /models/dense_pose && \
    chown -R ${USERNAME}:${USERNAME} /models
RUN mkdir -p /back_massage_bot && \
    chown -R ${USERNAME}:${USERNAME} /back_massage_bot
USER ${USERNAME}

# Create conda environment
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/environment.yml /back_massage_bot/environment.yml
RUN conda env create -f /back_massage_bot/environment.yml && \
    conda clean -a -y && \
    conda init bash


SHELL ["conda", "run", "-n", "back_massage_bot", "/bin/bash", "-c"]
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/post-entry-hooks.sh /back_massage_bot/post-entry-hooks.sh

# Install DensePose
RUN pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

RUN wget -O /models/dense_pose/Base-DensePose-RCNN-FPN.yaml \
https://raw.githubusercontent.com/facebookresearch/detectron2/refs/heads/main/projects/DensePose/configs/Base-DensePose-RCNN-FPN.yaml

RUN wget -O /models/dense_pose/densepose_rcnn_R_50_FPN_s1x.pkl \
https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl

RUN wget -O /models/dense_pose/Base-DensePose-RCNN-FPN-Human.yaml \
https://raw.githubusercontent.com/facebookresearch/detectron2/refs/heads/main/projects/DensePose/configs/cse/Base-DensePose-RCNN-FPN-Human.yaml

RUN wget -O /models/dense_pose/densepose_rcnn_R_50_FPN_s1x.yaml \
https://raw.githubusercontent.com/facebookresearch/detectron2/refs/heads/main/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml

RUN bash post-entry-hooks.sh

# Copy the rest of the project files
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot /back_massage_bot

# Install package
RUN pip install -r /back_massage_bot/requirements.txt && pip install -e /back_massage_bot

# Ensure conda environment is activated for interactive sessions
RUN echo "conda activate back_massage_bot" >> ~/.bashrc

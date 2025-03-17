FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ninja-build \
    build-essential \
    libopenblas-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxi6 \
    libxkbcommon-x11-0 \
    wget \
    tree \
    && rm -rf /var/lib/apt/lists/*


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

# Begin hack to install MinkowskiEngine with Cuda 12.1 (so that we can use Python 3.10 and then RoboStack for ROS Humble)
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/${USERNAME}/miniconda/bin:$PATH
RUN curl -sLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh \
    && chmod +x /tmp/miniconda.sh \
    && /tmp/miniconda.sh -b -p /home/${USERNAME}/miniconda \
    && rm /tmp/miniconda.sh

RUN conda init bash && \
echo "conda config --set auto_activate_base false" >> ~/.bashrc

# Create and set working directory with correct permissions
WORKDIR /back_massage_bot
USER root
RUN mkdir -p /models/ && \
chmod -R 777 /models && \
wget -O /models/human3d.ckpt https://omnomnom.vision.rwth-aachen.de/data/human3d/checkpoints/human3d.ckpt 
RUN mkdir -p /back_massage_bot && \
chmod -R 777 /back_massage_bot
USER ${USERNAME}

# Create conda environment
# COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/environment.yml /back_massage_bot/environment.yml
# RUN conda env create -f /back_massage_bot/environment.yml && \
RUN conda clean -a -y && \
conda init bash
RUN conda create -n back_massage_bot python=3.10


RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && sed -i '31i #include <thrust/execution_policy.h>' ./src/convolution_kernel.cuh \
    && sed -i '39i #include <thrust/unique.h>\n#include <thrust/remove.h>' ./src/coordinate_map_gpu.cu \
    && sed -i '38i #include <thrust/execution_policy.h>\n#include <thrust/reduce.h>\n#include <thrust/sort.h>' ./src/spmm.cu \
    && sed -i '38i #include <thrust/execution_policy.h>' ./src/3rdparty/concurrent_unordered_map.cuh \
    && python3 setup.py install --force_cuda --blas=openblas \
    && cd - \
    && rm -rf /tmp/MinkowskiEngine

# Copy package source code
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot /back_massage_bot
RUN cd /back_massage_bot/external/Human3D/third_party/pointnet2/ && python3 setup.py install
COPY --chown=${USERNAME}:${USERNAME} back_massage_bot/pyproject.toml /back_massage_bot/pyproject.toml

# Install package
RUN pip install -r /back_massage_bot/requirements.txt && pip install -e /back_massage_bot

# Ensure conda environment is activated for interactive sessions
RUN echo "conda activate back_massage_bot" >> ~/.bashrc

# RUN pip install pip==24.0 && pip install hydra-core==1.0.5
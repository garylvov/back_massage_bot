FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Make sure pip is up to date in the virtual environment
RUN pip3 install --upgrade pip

# Copy only requirements first
COPY therapeutech/requirements.txt /therapeutech/requirements.txt

# Install requirements (this will now be cached unless requirements.txt changes)
RUN pip3 install -r /therapeutech/requirements.txt

# Copy the rest of the project files
COPY therapeutech /therapeutech

# Install package
RUN pip3 install -e /therapeutech

# Ensure virtualenv is activated for interactive sessions
RUN echo "source $VIRTUAL_ENV/bin/activate" >> ~/.bashrc

# Set working directory
WORKDIR /therapeutech

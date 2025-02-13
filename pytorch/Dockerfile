# Use NVIDIA NGC PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Set working directory
WORKDIR /app

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh

# Set environment variables for Conda
ENV PATH="/opt/conda/bin:$PATH"

# Copy the environment YAML file
COPY myenv.yml /app/myenv.yml

# Create the conda environment
RUN conda env create -f /app/myenv.yml && \
    conda clean --all -y

# Set the default Conda environment
ENV PATH="/opt/conda/envs/myenv/bin:$PATH"
ENV CONDA_DEFAULT_ENV=myenv

# Ensure the correct shell is used to activate Conda environments
SHELL ["/bin/bash", "-c"]

# Default command
CMD ["python", "main.py"]


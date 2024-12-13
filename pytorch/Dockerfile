# Use NVIDIA NGC PyTorch image
FROM nvcr.io/nvidia/pytorch:24.06-py3

# Set working directory
WORKDIR /app

# Copy the application code and requirements
COPY requirements.txt /app/requirements.txt
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Default command
CMD ["python", "main.py"]

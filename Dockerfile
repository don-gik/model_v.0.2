# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the remaining Python dependencies
# PyTorch, torchvision, and torchaudio are already in the base image, 
# so pip will skip them and only install the other packages.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the default command to open a bash shell
CMD ["/bin/bash"]

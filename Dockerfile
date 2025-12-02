# 1. Base Image: Use standard Python Slim (Size: ~150MB)
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files (Saves space)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install System Tools (Git + OpenCV dependencies)
# We combine these into one RUN command to reduce "Layer Bloat"
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Set Working Directory
WORKDIR /app

# 5. Install PyTorch FIRST (The Heavy Hitter)
# We specify the CUDA 12.1 index directly to get the optimized wheels
# This installs Torch (~2.3GB) instead of the generic bloated version
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Install Remaining Dependencies
COPY requirements.txt .
# We remove 'torch' and 'torchvision' from requirements.txt on the fly
# to prevent pip from reinstalling/downgrading them.
RUN sed -i '/torch/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Project (Last step to optimize caching)
COPY . .

# 8. Git Safe Directory
RUN git config --global --add safe.directory /app

CMD ["/bin/bash"]
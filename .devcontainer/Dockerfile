# .devcontainer/Dockerfile
FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

# System dependencies useful for image processing and builds
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    libjpeg-dev zlib1g-dev libpng-dev libtiff5-dev libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the vscode user provided by the base image
USER vscode
WORKDIR /workspaces/app

# Upgrade pip tooling (packages installed later via postCreateCommand or requirements.txt)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Optional: pre-install a minimal ML/tooling set so the first open is faster
RUN pip install --no-cache-dir \
    numpy scipy scikit-image scikit-learn opencv-python-headless \
    albumentations tqdm matplotlib seaborn jupyter ipykernel ipywidgets \
    mlflow dvc dvc[gdrive] \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

CMD [ "bash" ]


# Można dodać NVIDIA CUDA jeśli potrzebne
# Base image with CUDA 11.3.0 and Ubuntu 20.04
FROM nvidia/cuda:11.3.0-base-ubuntu20.04

# Install python3 and pip
RUN apt update && apt install -y python3-pip
# Install opencv dependencies
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libsm6 libxext6 libxrender-dev ffmpeg

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY requirements.txt /app
RUN pip install -r requirements.txt

ENTRYPOINT bash
FROM nvidia/cuda:11.3.0-base-ubuntu20.04

RUN apt update && apt install -y python3-pip
# Install opencv dependencies
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends libsm6 libxext6 libxrender-dev ffmpeg

# Set the working directory to /app
WORKDIR /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

ENTRYPOINT bash
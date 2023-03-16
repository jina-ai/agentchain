# Deploying with docker

Dependency management can be challenging, but fortunately there is a solution: Docker. Using Docker to deploy AgentChain offers an easy and reproducible way to get started. Since the agents are deployed on GPU, you will need the NVIDIA Container Toolkit.

If you've never used Docker with a GPU before, follow the Toolkit installation instructions:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit

## Building the image

Building the image is pretty straightforward

```bash
cp ./requirements.txt ./docker
cd docker
docker build -t agentchain .
cd ..
```

## Download model checkpoints

```bash
bash download.sh
```

The model checkpoints are 44GB in total so this can take a while.

## Run container

```bash
docker run --name agentchain -it -v $(pwd):/app --gpus all -p 7861:7861 agentchain
```

## Set env variable and start server

You will now be in a bash shell. Here you need to export the API keys as environment variable for the server. The Open AI API key and the Serp API key are required as they power the main agent and the search agent respectively.

```bash
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
SERPAPI_API_KEY=<YOUR_SERPAPI_API_KEY>
```

(Optional) If you want the CommsAgent to be able to make phone calls you will need to export a few more variables. The AWS_S3_BUCKET_NAME specified needs to be a public access bucket.

```bash
AWS_ACCESS_KEY_ID={YOUR_AWS_ACCESS_KEY_ID}
AWS_SECRET_ACCESS_KEY={YOUR_AWS_SECRET_ACCESS_KEY}
TWILIO_ACCOUNT_SID={YOUR_TWILIO_ACCOUNT_SID}
TWILIO_AUTH_TOKEN={YOUR_TWILIO_AUTH_TOKEN}
AWS_S3_BUCKET_NAME={YOUR_AWS_S3_BUCKET_NAME}
```

You can now start the server by running the main script.

```bash
python3 main.py
```

The server may take about an hour before serving the first time as there are a few more model checkpoints to install. The installs may also timeout the first time in which case, you can run `python [main.py](http://main.py)` again to resume downloading checkpoints.
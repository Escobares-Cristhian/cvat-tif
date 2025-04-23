docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build
cd serverless/ ; bash deploy_gpu.sh pytorch/facebookresearch/sam/nuclio/ ; cd ../

docker compose down --remove-orphans
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d
cd serverless/ ; bash deploy_gpu.sh pytorch/facebookresearch/sam/nuclio/ ; cd ../

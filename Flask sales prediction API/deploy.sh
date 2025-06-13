#!/bin/bash

# Deployment script for XGBoost Flask API
# Based on the provided deployment guide

set -e  # Exit on any error

# Configuration
IMAGE_NAME="xgboost-flask-api"
IMAGE_TAG="latest"

echo "🚀 Starting deployment process for XGBoost Flask API"
echo "=================================================="

# Check if required files exist
echo "📋 Checking required files..."
required_files=("app.py" "requirements.txt" "Dockerfile" "XGBRegressor.pkl")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: Required file '$file' not found!"
        echo "Please ensure all files are in the current directory."
        exit 1
    fi
    echo "✅ Found: $file"
done

echo ""
echo "🔨 Building Docker image..."

# Detect platform and build accordingly
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected - Building for x86 architecture (AWS compatible)"
    docker buildx create --use 2>/dev/null || true
    docker buildx build --platform linux/amd64 -t ${IMAGE_NAME}:${IMAGE_TAG} . --load
else
    echo "🐧 Linux/Windows detected - Building for native architecture"
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
fi

echo "✅ Docker image built successfully!"

echo ""
echo "🧪 Testing the Docker container locally..."

# Stop any existing container
docker stop ${IMAGE_NAME} 2>/dev/null || true
docker rm ${IMAGE_NAME} 2>/dev/null || true

# Run the container
echo "Starting container on port 5000..."
docker run -d -p 5000:5000 --name ${IMAGE_NAME} ${IMAGE_NAME}:${IMAGE_TAG}

# Wait for container to start
echo "Waiting for container to start..."
sleep 10

# Test the API
echo "Testing API health endpoint..."
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ API is responding!"
    echo "🔗 You can test the API at: http://localhost:5000"
    echo ""
    echo "📖 Example test with curl:"
    echo 'curl -X POST http://localhost:5000/predict \'
    echo '  -H "Content-Type: application/json" \'
    echo '  -d "{\"store_ID\":49,\"day_of_week\":4,\"date\":\"26/06/2014\",\"nb_customers_on_day\":1254,\"open\":1,\"promotion\":0,\"state_holiday\":\"0\",\"school_holiday\":1}"'
else
    echo "❌ API health check failed!"
    echo "Container logs:"
    docker logs ${IMAGE_NAME}
    exit 1
fi

echo ""
echo "🏗️  Deployment Summary:"
echo "======================"
echo "✅ Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "✅ Container running on: http://localhost:5000"
echo "✅ Health check: http://localhost:5000/health"
echo "✅ Main endpoint: http://localhost:5000/predict"

echo ""
echo "📝 Next Steps for AWS Deployment:"
echo "================================="
echo "1. Configure AWS CLI: aws configure"
echo "2. Create ECR repository: Follow AWS ECR console instructions"
echo "3. Push image to ECR: Use 'View push commands' in ECR console"
echo "4. Create EC2 instance: Follow the deployment guide"
echo "5. Run on EC2: docker run -d -p 5000:5000 [ECR_IMAGE_URI]"

echo ""
echo "🛑 To stop the local container:"
echo "docker stop ${IMAGE_NAME} && docker rm ${IMAGE_NAME}"

echo ""
echo "🎉 Local deployment completed successfully!"
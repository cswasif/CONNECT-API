#!/bin/bash
set -e

echo "========================================="
echo "ConnectAPI - Deployment Script"
echo "========================================="

APP_DIR="/home/ubuntu/zconnapi2-main"
SERVICE_NAME="connectapi"

cd $APP_DIR

# Pull latest changes (if using git)
# git pull origin main

echo "Installing/updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo "Running health check before restart..."
HEALTH_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")

echo "Restarting service..."
sudo systemctl restart $SERVICE_NAME

echo "Waiting for service to start..."
sleep 5

echo "Checking service status..."
sudo systemctl status $SERVICE_NAME --no-pager || true

echo "Running health check..."
for i in {1..10}; do
    HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
    if [ "$HEALTH" = "200" ]; then
        echo "✓ Health check passed!"
        echo "========================================="
        echo "Deployment completed successfully!"
        echo "========================================="
        exit 0
    fi
    echo "Waiting for service to be healthy... ($i/10)"
    sleep 2
done

echo "✗ Health check failed!"
echo "Rolling back..."
sudo systemctl restart $SERVICE_NAME
echo "Check logs: sudo journalctl -u $SERVICE_NAME -n 50"
exit 1

#!/bin/bash
set -e

echo "========================================="
echo "ConnectAPI - Production Setup Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
APP_DIR="/home/ubuntu/zconnapi2-main"
VENV_DIR="$APP_DIR/venv"
SERVICE_NAME="connectapi"
NGINX_CONFIG="/etc/nginx/sites-available/connectapi"
LOG_DIR="/var/log/connectapi"
RUN_DIR="/var/run/connectapi"

echo -e "${GREEN}Step 1: Installing system dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv nginx

echo -e "${GREEN}Step 2: Creating directories...${NC}"
sudo mkdir -p $LOG_DIR
sudo mkdir -p $RUN_DIR
sudo chown ubuntu:ubuntu $LOG_DIR
sudo chown ubuntu:ubuntu $RUN_DIR

echo -e "${GREEN}Step 3: Setting up Python virtual environment...${NC}"
cd $APP_DIR
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

echo -e "${GREEN}Step 4: Installing Python dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}Step 5: Creating environment file...${NC}"
if [ ! -f "$APP_DIR/.env" ]; then
    cp env.template .env
    echo -e "${YELLOW}WARNING: Please edit .env file with your configuration${NC}"
    echo -e "${YELLOW}File location: $APP_DIR/.env${NC}"
else
    echo ".env file already exists"
fi

echo -e "${GREEN}Step 6: Installing systemd service...${NC}"
sudo cp connectapi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable $SERVICE_NAME

echo -e "${GREEN}Step 7: Configuring Nginx...${NC}"
sudo cp nginx-connectapi.conf $NGINX_CONFIG
if [ ! -L "/etc/nginx/sites-enabled/connectapi" ]; then
    sudo ln -s $NGINX_CONFIG /etc/nginx/sites-enabled/
fi

# Test Nginx configuration
sudo nginx -t

echo -e "${GREEN}Step 8: Starting services...${NC}"
sudo systemctl restart $SERVICE_NAME
sudo systemctl restart nginx

echo ""
echo "========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit $APP_DIR/.env with your configuration"
echo "2. Restart the service: sudo systemctl restart $SERVICE_NAME"
echo "3. Check status: sudo systemctl status $SERVICE_NAME"
echo "4. View logs: sudo journalctl -u $SERVICE_NAME -f"
echo "5. Configure Cloudflare Tunnel for connect.routinez.app"
echo ""
echo "Useful commands:"
echo "  Start service:   sudo systemctl start $SERVICE_NAME"
echo "  Stop service:    sudo systemctl stop $SERVICE_NAME"
echo "  Restart service: sudo systemctl restart $SERVICE_NAME"
echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
echo "  Check health:    curl http://localhost:8000/health"
echo ""

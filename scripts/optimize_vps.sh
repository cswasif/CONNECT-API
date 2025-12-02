#!/bin/bash

# ConnectAPI - Optimization Script for Micro Instance
# Fixes high load and memory issues

echo "========================================"
echo "   Optimizing VPS for Micro Instance"
echo "========================================"

# 1. Stop Services to free up resources immediately
echo "[1/4] Stopping services..."
sudo systemctl stop connectapi
sudo systemctl stop redis-server
sudo systemctl stop nginx
sudo systemctl stop cloudflared

# 2. Add Swap Space (2GB)
echo "[2/4] Adding 2GB Swap Space..."
if [ ! -f /swapfile ]; then
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    echo "Swap created successfully."
else
    echo "Swap file already exists."
fi

# 3. Optimize Gunicorn (Reduce workers)
echo "[3/4] Optimizing Gunicorn..."
# Replace the workers line in gunicorn.conf.py
sed -i 's/workers = .*/workers = 2/g' /home/ubuntu/zconnapi2-main/gunicorn.conf.py
echo "Reduced Gunicorn workers to 2."

# 4. Optimize Redis (Reduce memory)
echo "[4/4] Optimizing Redis..."
# Update maxmemory in redis.conf
if grep -q "maxmemory " /etc/redis/redis.conf; then
    sudo sed -i 's/maxmemory .*/maxmemory 128mb/g' /etc/redis/redis.conf
else
    echo "maxmemory 128mb" | sudo tee -a /etc/redis/redis.conf
fi
echo "Reduced Redis memory limit to 128MB."

# 5. Restart Services
echo "========================================"
echo "   Restarting Services..."
echo "========================================"
sudo systemctl start redis-server
sudo systemctl start connectapi
sudo systemctl start nginx
sudo systemctl start cloudflared

# 6. Verify
echo "========================================"
echo "   Status Check"
echo "========================================"
free -h
echo "----------------------------------------"
uptime
echo "----------------------------------------"
sudo systemctl status connectapi --no-pager | grep Active

echo "Optimization Complete!"

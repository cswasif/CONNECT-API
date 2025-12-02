#!/bin/bash
set -e

echo "========================================="
echo "ConnectAPI - Redis Installation Script"
echo "========================================="

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install Redis
echo "Installing Redis..."
sudo apt-get install -y redis-server

# Configure Redis
echo "Configuring Redis..."
sudo sed -i 's/^supervised no/supervised systemd/' /etc/redis/redis.conf
sudo sed -i 's/^bind 127.0.0.1 ::1/bind 127.0.0.1/' /etc/redis/redis.conf

# Set memory limit (adjust as needed)
echo "maxmemory 512mb" | sudo tee -a /etc/redis/redis.conf
echo "maxmemory-policy allkeys-lru" | sudo tee -a /etc/redis/redis.conf

# Enable and start Redis
echo "Starting Redis service..."
sudo systemctl enable redis-server
sudo systemctl restart redis-server

# Check Redis status
echo ""
echo "Checking Redis status..."
sudo systemctl status redis-server --no-pager

# Test Redis connection
echo ""
echo "Testing Redis connection..."
redis-cli ping

echo ""
echo "========================================="
echo "Redis installation completed!"
echo "========================================="
echo "Redis is running on localhost:6379"
echo "To check status: sudo systemctl status redis-server"
echo "To view logs: sudo journalctl -u redis-server -f"

# ConnectAPI - RoutineZ Backend

High-performance FastAPI backend for the RoutineZ application, deployed on Oracle Cloud Infrastructure (OCI) with Cloudflare Tunnel for secure, zero-trust access.

## 🏗 Architecture

```mermaid
graph LR
    User[User] -->|HTTPS| CF[Cloudflare Edge]
    CF -->|Tunnel| Cloudflared[Cloudflared (VPS)]
    Cloudflared -->|HTTP| Nginx[Nginx Reverse Proxy]
    Nginx -->|Proxy| Gunicorn[Gunicorn (2 Workers)]
    Gunicorn -->|ASGI| FastAPI[FastAPI App]
    FastAPI -->|Async| Redis[Local Redis (Cache/DB)]
```

### Tech Stack
- **Framework**: FastAPI (Python 3.8+)
- **Server**: Gunicorn + Uvicorn Workers
- **Proxy**: Nginx
- **Database**: Redis (Local, Asyncio)
- **Infrastructure**: Oracle Cloud VM (Ubuntu 20.04)
- **Networking**: Cloudflare Tunnel (No open inbound ports)

## 📂 Project Structure

```
zconnapi2-main/
├── config/                 # Configuration files
│   ├── nginx-connectapi.conf   # Nginx site config
│   ├── connectapi.service      # Systemd service
│   └── cloudflared-config.yml  # Tunnel config
├── scripts/                # Maintenance scripts
│   ├── setup.sh                # Initial VPS setup
│   ├── install-redis.sh        # Redis installation
│   └── optimize_vps.sh         # Performance tuning (Swap, Limits)
├── main.py                 # Application entry point
├── auth_config.py          # Auth settings
├── gunicorn.conf.py        # Gunicorn config
├── deploy.sh               # Deployment script
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start (Local Development)

1.  **Clone & Setup**:
    ```bash
    git clone <repo>
    cd zconnapi2-main
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  **Configure Environment**:
    ```bash
    cp env.template .env
    # Edit .env with your credentials
    ```

3.  **Run Server**:
    ```bash
    uvicorn main:app --reload
    ```

## ☁️ Production Deployment

The application is designed to run on a VPS managed by Systemd.

### Initial Setup
Run the setup script to install dependencies, Redis, and configure services:
```bash
./scripts/setup.sh
```

### Deploy Updates
To pull the latest code and restart services:
```bash
./deploy.sh
```

### Maintenance
- **Check Status**: `sudo systemctl status connectapi`
- **View Logs**: `sudo journalctl -u connectapi -f`
- **Restart**: `sudo systemctl restart connectapi`
- **Redis Monitor**: `redis-cli monitor`

## 🔧 Configuration

### Environment Variables (`.env`)
| Variable | Description | Default |
| :--- | :--- | :--- |
| `REDIS_HOST` | Redis Hostname | `localhost` |
| `REDIS_PORT` | Redis Port | `6379` |
| `WORKERS` | Gunicorn Workers | `2` |
| `DEBUG_MODE` | Enable Debug Logs | `false` |

### Performance Tuning
The VPS is optimized for low-memory environments (1GB RAM):
- **Swap**: 2GB Swap file enabled.
- **Redis**: Max memory limited to 128MB (`allkeys-lru`).
- **Gunicorn**: Limited to 2 workers to prevent OOM.

## 🔗 API Endpoints
- `GET /health`: Health check (Redis connection status).
- `GET /ready`: Readiness check.
- `POST /enter-tokens`: Authenticate user.
- `GET /raw-schedule`: Fetch schedule.

## 🛠 Troubleshooting

### Common Issues

1.  **Service Won't Start**:
    ```bash
    sudo journalctl -u connectapi -n 50
    ```
    Check for Python errors or missing environment variables.

2.  **Public Access Fails (1033 Error)**:
    Cloudflare Tunnel cannot reach the service.
    - Check if Nginx is running: `sudo systemctl status nginx`
    - Check if Cloudflared is running: `sudo systemctl status cloudflared`
    - Verify ingress rules in `/etc/cloudflared/config.yml`.

3.  **Redis Connection Error**:
    - Check if Redis is running: `sudo systemctl status redis-server`
    - Verify memory usage: `redis-cli info memory`
    - Check logs: `sudo tail /var/log/redis/redis-server.log`

### Backup & Recovery

**Backup Redis Data**:
```bash
sudo cp /var/lib/redis/dump.rdb /home/ubuntu/backups/redis-$(date +%F).rdb
```

**Backup Configuration**:
```bash
tar -czf config-backup.tar.gz .env config/
```
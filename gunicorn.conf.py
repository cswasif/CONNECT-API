import multiprocessing
import os

# Server Socket
bind = f"{os.getenv('HOST', '127.0.0.1')}:{os.getenv('PORT', '8000')}"
backlog = 2048

# Worker Processes
workers = int(os.getenv('WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'uvicorn.workers.UvicornWorker'
worker_connections = 1000
max_requests = 10000
max_requests_jitter = 1000
timeout = 120
keepalive = 5

# Logging
accesslog = '/var/log/connectapi/access.log'
errorlog = '/var/log/connectapi/error.log'
loglevel = os.getenv('LOG_LEVEL', 'info').lower()
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process Naming
proc_name = 'connectapi'

# Server Mechanics
daemon = False
pidfile = '/var/run/connectapi/connectapi.pid'
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
# keyfile = '/path/to/key.pem'
# certfile = '/path/to/cert.pem'

# Graceful Shutdown
graceful_timeout = 30

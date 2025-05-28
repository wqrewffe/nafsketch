import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes (set based on CPU count)
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class
worker_class = 'sync'  # or 'gevent' if you want async support

# Connections (mostly for async workers)
worker_connections = 10000  # Very high for async; not used in 'sync'

# Timeout settings
timeout = 3600  # 1 hour timeout for long-running requests
keepalive = 5   # keep connections alive longer for reuse

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'nafsketch'

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (optional)
keyfile = None
certfile = None

# =============================================================
# ConnectAPI - BRACU Connect API Server
# Developed by Wasif Faisal to support https://routinez.app/
# =============================================================

from fastapi import FastAPI, Response, HTTPException, Header, Request, Cookie, status, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse
import httpx
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel
import secrets
from urllib.parse import urlencode, urlparse
from starlette.middleware.sessions import SessionMiddleware
from auth_config import settings
import logging
import traceback
from datetime import datetime
import json
import hashlib
import base64
import os
import time
import redis.asyncio as aioredis
import jwt
from dotenv import load_dotenv
from collections import defaultdict
from asyncio import Lock
import asyncio
from fastapi.concurrency import run_in_threadpool
from contextlib import asynccontextmanager
from functools import wraps
from weakref import WeakSet
import sys

# Load environment variables
load_dotenv()

# Global Debug Control
# Set these flags to control debugging behavior
DEBUG_MODE = False  # Set to True to enable debug logging and prints
TRACE_MODE = False  # Set to True to enable detailed tracing (stack traces, etc)
SHOW_VIEW_TOKENS = True  # Set to True to allow anyone to view tokens

# Get OAuth2 credentials from environment variables
OAUTH_CLIENT_ID = os.getenv('OAUTH_CLIENT_ID', 'connect-portal')
OAUTH_CLIENT_SECRET = os.getenv('OAUTH_CLIENT_SECRET', '')
OAUTH_TOKEN_URL = os.getenv('OAUTH_TOKEN_URL', 'https://sso.bracu.ac.bd/realms/bracu/protocol/openid-connect/token')

# Development mode - set to True for local development
DEV_MODE = os.getenv('DEV_MODE', 'false').lower() == 'true'

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Set httpx logging to WARNING level to suppress INFO messages
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def debug_print(*args, **kwargs):
    """Debug print function that only prints in DEBUG_MODE"""
    if DEBUG_MODE:
        print("[DEBUG]", *args, **kwargs)
        # Also log to file in debug mode
        logger.debug(" ".join(str(arg) for arg in args))
    elif TRACE_MODE:
        # In trace mode, still log but don't print
        logger.debug(" ".join(str(arg) for arg in args))

def trace_print(*args, **kwargs):
    """Trace print function that only prints in TRACE_MODE"""
    if TRACE_MODE:
        print("[TRACE]", *args, **kwargs)
        logger.debug("[TRACE] " + " ".join(str(arg) for arg in args))

# Record server start time
start_time = time.time()
debug_print(f"Server starting at {datetime.fromtimestamp(start_time).isoformat()}")

# Add after other constants
REQUEST_DEBOUNCE_WINDOW = 2  # seconds

# Add before the raw_schedule endpoint
last_request_time = defaultdict(float)

# Add after other imports
from asyncio import Lock

# Add after other constants
schedule_locks = {}

# Add after other imports
import asyncio
from typing import Optional, Dict, Any, Tuple

# Add after other constants
BACKGROUND_TASKS = {}
TASK_LOCKS = {}
ACTIVE_TASKS = WeakSet()  # Keep track of active tasks

# Add after other constants
REDIS_RETRY_COUNT = 3
REDIS_RETRY_DELAY = 1  # seconds

# Add after other constants
TASK_RETENTION_TIME = 300  # Keep completed tasks for 5 minutes

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = datetime.now().isoformat()

# Global Redis connection pool
redis_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    global redis_pool
    
    # Startup
    logger.info("Starting ConnectAPI...")
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_password = os.getenv('REDIS_PASSWORD', None)
    max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', 50))
    
    try:
        redis_pool = aioredis.ConnectionPool(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password if redis_password else None,
            max_connections=max_connections,
            decode_responses=True,
            health_check_interval=30
        )
        # Test connection
        test_client = aioredis.Redis(connection_pool=redis_pool)
        await test_client.ping()
        logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        await test_client.close()
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down ConnectAPI...")
    if redis_pool:
        await redis_pool.disconnect()
        logger.info("Redis connection pool closed")

# Create FastAPI app with lifespan management
app = FastAPI(
    title="ConnectAPI",
    description="BRACU Connect API Server",
    version="2.0.0",
    lifespan=lifespan
)

# Get allowed origins from environment
allowed_origins_str = os.getenv('ALLOWED_ORIGINS', 'https://routinez.app,https://routinez.vercel.app')
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',')]
if DEV_MODE:
    allowed_origins = ["*"]

# Add middleware in correct order
session_secret = os.getenv('SESSION_SECRET', os.getenv('SECRET_KEY', 'super-secret-session-key-change-in-production'))
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret,
    session_cookie="connectapi_session",
    max_age=1800,  # 30 minutes
    same_site="lax",
    https_only=not DEV_MODE,  # HTTPS only in production
    path="/"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add debug control endpoints
@app.get("/debug/status")
async def debug_status():
    """Get current debug status"""
    return {
        "debug_mode": DEBUG_MODE,
        "trace_mode": TRACE_MODE,
        "log_level": "DEBUG" if DEBUG_MODE else "INFO"
    }

@app.post("/debug/toggle")
async def toggle_debug(mode: str = Query(..., regex="^(debug|trace)$")):
    """Toggle debug or trace mode"""
    global DEBUG_MODE, TRACE_MODE
    
    if mode == "debug":
        DEBUG_MODE = not DEBUG_MODE
        # Update log level
        logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
        debug_print(f"Debug mode {'enabled' if DEBUG_MODE else 'disabled'}")
        return {"debug_mode": DEBUG_MODE}
    else:  # trace
        TRACE_MODE = not TRACE_MODE
        trace_print(f"Trace mode {'enabled' if TRACE_MODE else 'disabled'}")
        return {"trace_mode": TRACE_MODE}

# Health check endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring and load balancers"""
    try:
        redis_conn = await get_redis()
        await redis_conn.ping()
        await redis_conn.close()
        
        return {
            "status": "healthy",
            "service": "ConnectAPI",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat(),
            "redis": "connected"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "service": "ConnectAPI",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint - checks if service is ready to accept traffic"""
    try:
        # Check Redis connection
        redis_conn = await get_redis()
        await redis_conn.ping()
        
        # Check if we can read from Redis
        await redis_conn.get("test_key")
        await redis_conn.close()
        
        return {
            "status": "ready",
            "service": "ConnectAPI",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "redis_connection": "ok",
                "redis_operations": "ok"
            }
        }
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "ConnectAPI",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Session initialization will be handled by the root endpoint and token endpoints

# Redis Configuration - Now using local Redis
async def get_redis():
    """Get Redis client from connection pool"""
    global redis_pool
    if redis_pool is None:
        raise RuntimeError("Redis connection pool not initialized")
    return aioredis.Redis(connection_pool=redis_pool)

def get_redis_sync():
    """Synchronous Redis client for compatibility (deprecated, use async version)"""
    import redis
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_db = int(os.getenv('REDIS_DB', 0))
    redis_password = os.getenv('REDIS_PASSWORD', None)
    
    return redis.Redis(
        host=redis_host,
        port=redis_port,
        db=redis_db,
        password=redis_password if redis_password else None,
        decode_responses=True
    )

# Async Redis operations - now truly async with connection pool
async def redis_get(redis_conn, key):
    """Async Redis get operation."""
    return await redis_conn.get(key)

async def redis_set(redis_conn, key, value, ex=None):
    """Async Redis set operation."""
    if ex is not None:
        return await redis_conn.set(key, value, ex=ex)
    else:
        return await redis_conn.set(key, value)

async def redis_keys(redis_conn, pattern):
    """Async Redis keys operation."""
    return await redis_conn.keys(pattern)

async def redis_delete(redis_conn, key):
    """Async Redis delete operation."""
    return await redis_conn.delete(key)

# Sync versions for compatibility
def redis_get_sync(redis_conn, key):
    """Synchronous Redis get operation."""
    return redis_conn.get(key)

def redis_set_sync(redis_conn, key, value, ex=None):
    """Synchronous Redis set operation."""
    if ex is not None:
        return redis_conn.set(key, value, ex=ex)
    else:
        return redis_conn.set(key, value)

def redis_keys_sync(redis_conn, pattern):
    """Synchronous Redis keys operation."""
    return redis_conn.keys(pattern)

def redis_delete_sync(redis_conn, key):
    """Synchronous Redis delete operation."""
    return redis_conn.delete(key)

def decode_jwt_token(token: str) -> dict:
    """Decode a JWT token without verification to get expiration time."""
    try:
        # Split the token and get the payload part (second part)
        parts = token.split('.')
        if len(parts) != 3:
            logger.error("Invalid JWT token format")
            return {}
        
        # Decode the payload
        # Add padding if needed
        padding = len(parts[1]) % 4
        if padding:
            parts[1] += '=' * (4 - padding)
        
        payload = json.loads(base64.b64decode(parts[1]).decode('utf-8'))
        return payload
    except Exception as e:
        logger.error(f"Error decoding JWT token: {str(e)}")
        return {}

def with_redis_retry(func):
    """Decorator to retry Redis operations with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(REDIS_RETRY_COUNT):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < REDIS_RETRY_COUNT - 1:  # Don't sleep on last attempt
                    time.sleep(REDIS_RETRY_DELAY * (2 ** attempt))
                continue
        logger.error(f"Redis operation failed after {REDIS_RETRY_COUNT} attempts: {str(last_error)}")
        raise last_error
    return wrapper

@with_redis_retry
def _save_tokens_to_redis_sync(session_id, tokens):
    """Save tokens to Redis with proper expiration time from JWT."""
    try:
        redis_conn = get_redis_sync()
        now = int(time.time())
        
        # Get expiration from JWT if we have an access token
        if "access_token" in tokens:
            jwt_data = decode_jwt_token(tokens["access_token"])
            if "exp" in jwt_data:
                tokens["expires_at"] = jwt_data["exp"]
                tokens["expires_in"] = max(0, jwt_data["exp"] - now)
                debug_print(f"Token expiration from JWT: {tokens['expires_in']} seconds remaining")
            elif "expires_in" in tokens:
                # Use the actual expires_in from SSO response
                tokens["expires_at"] = now + int(tokens["expires_in"])
                debug_print(f"Token expiration from SSO: {tokens['expires_in']} seconds")
            else:
                # Default to 5 minutes based on actual token data
                tokens["expires_at"] = now + 300
                tokens["expires_in"] = 300
                logger.warning("No expiration found, using default 5 minutes")
        
        # Set refresh token expiration from SSO response or default to 30 minutes
        if "refresh_token" in tokens:
            if "refresh_expires_in" in tokens:
                tokens["refresh_expires_at"] = now + int(tokens["refresh_expires_in"])
                debug_print(f"Refresh token expiration from SSO: {tokens['refresh_expires_in']} seconds")
            else:
                tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes based on actual data
                debug_print("Refresh token expiration set to 30 minutes")
        
        # Save tokens with expiration
        key = f"tokens:{session_id}"
        redis_conn.set(key, json.dumps(tokens))
        
        # Set key expiration to match the refresh token expiration
        refresh_expires_in = tokens.get("refresh_expires_in", 30 * 60)
        redis_conn.expire(key, int(refresh_expires_in))  # Use actual refresh token lifetime
        
        debug_print(f"Tokens saved in Redis for session {session_id}. Access token expires in {tokens.get('expires_in', 0)}s")
        return True
    except Exception as e:
        logger.error(f"Error saving tokens to Redis: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        raise

async def save_tokens_to_redis(session_id, tokens):
    """Async wrapper for save_tokens_to_redis function."""
    try:
        # Use sync version since we're using sync Redis client
        return _save_tokens_to_redis_sync(session_id, tokens)
    except Exception as e:
        logger.error(f"Error in async save_tokens_to_redis: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        raise

@with_redis_retry
def _load_tokens_from_redis_sync(session_id):
    """Load tokens from Redis with validation and refresh if needed."""
    try:
        redis_conn = get_redis_sync()
        key = f"tokens:{session_id}"
        data = redis_conn.get(key)
        
        if data:
            tokens = json.loads(data)
            now = int(time.time())
            
            # Check if token needs proactive refresh (50% through its lifetime for 5-min tokens)
            needs_refresh = False
            if "expires_at" in tokens and "refresh_token" in tokens:
                total_lifetime = tokens["expires_at"] - (tokens.get("activated_at") or (tokens["expires_at"] - 300))
                time_elapsed = now - (tokens.get("activated_at") or (tokens["expires_at"] - 300))
                if time_elapsed >= (total_lifetime * 0.5):  # Refresh at 50% for 5-min tokens
                    needs_refresh = True
                    debug_print(f"Token needs proactive refresh for session {session_id} (50% lifetime elapsed)")
            
            # If token is not expired and doesn't need proactive refresh
            if not is_token_expired(tokens) and not needs_refresh:
                debug_print(f"Valid tokens loaded from Redis for session {session_id}")
                return tokens
            
            # Try to refresh the token if we have a refresh token
            if "refresh_token" in tokens:
                try:
                    debug_print(f"Attempting token refresh for session {session_id}")
                    new_tokens = refresh_access_token_sync(tokens["refresh_token"])
                    if new_tokens:
                        new_tokens["activated_at"] = now
                        _save_tokens_to_redis_sync(session_id, new_tokens)
                        return new_tokens
                except Exception as e:
                    logger.error(f"Token refresh failed (sync): {str(e)}")
                    if DEV_MODE:
                        logger.error(traceback.format_exc())
            
            # If we get here, tokens are expired/invalid and refresh failed
            redis_conn.delete(key)
            return None
        else:
            debug_print(f"No tokens found in Redis for session {session_id}")
            return None
    except Exception as e:
        logger.error(f"Error loading tokens from Redis: {str(e)}")
        if DEV_MODE:
            logger.error(traceback.format_exc())
        return None

async def load_tokens_from_redis(session_id):
    """Async wrapper for load_tokens_from_redis function with token refresh."""
    try:
        redis_conn = get_redis_sync()
        key = f"tokens:{session_id}"
        data = redis_conn.get(key)
        
        if data:
            tokens = json.loads(data)
            now = int(time.time())
            
            # Check if token needs proactive refresh (75% through its lifetime)
            needs_refresh = False
            if "expires_at" in tokens and "refresh_token" in tokens:
                total_lifetime = tokens["expires_at"] - (tokens.get("activated_at") or (tokens["expires_at"] - 600))
                time_elapsed = now - (tokens.get("activated_at") or (tokens["expires_at"] - 600))
                if time_elapsed >= (total_lifetime * 0.75):
                    needs_refresh = True
                    debug_print(f"Token needs proactive refresh for session {session_id} (75% lifetime elapsed)")
            
            # If token is not expired and doesn't need proactive refresh
            if not is_token_expired(tokens) and not needs_refresh:
                debug_print(f"Valid tokens loaded from Redis for session {session_id}")
                return tokens
            
            # Try to refresh the token if we have a refresh token
            if "refresh_token" in tokens:
                try:
                    debug_print(f"Attempting token refresh for session {session_id}")
                    new_tokens = await refresh_access_token(tokens["refresh_token"])
                    if new_tokens:
                        new_tokens["activated_at"] = now
                        await save_tokens_to_redis(session_id, new_tokens)
                        return new_tokens
                except Exception as e:
                    logger.error(f"Token refresh failed: {str(e)}")
                    if DEV_MODE:
                        logger.error(traceback.format_exc())
            
            # If we get here, tokens are expired/invalid and refresh failed
            redis_conn.delete(key)
            return None
        else:
            debug_print(f"No tokens found in Redis for session {session_id}")
            return None
    except Exception as e:
        logger.error(f"Error loading tokens from Redis: {str(e)}")
        if DEV_MODE:
            logger.error(traceback.format_exc())
        return None

@with_redis_retry
def _save_global_student_tokens_sync(student_id, tokens):
    try:
        redis_conn = get_redis_sync()
        if "expires_in" in tokens:
            tokens["expires_at"] = int(time.time()) + int(tokens["expires_in"])
        redis_conn.set(f"student_tokens:{student_id}", json.dumps(tokens))
        logger.info(f"Global tokens updated for student_id {student_id}.")
        return True
    except Exception as e:
        logger.error(f"Error saving global student tokens to Redis: {str(e)}")
        raise

async def save_global_student_tokens(student_id, tokens):
    try:
        return _save_global_student_tokens_sync(student_id, tokens)
    except Exception as e:
        logger.error(f"Error in async save_global_student_tokens: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        raise

@with_redis_retry
def _load_global_student_tokens_sync(student_id):
    try:
        redis_conn = get_redis_sync()
        data = redis_conn.get(f"student_tokens:{student_id}")
        if data:
            tokens = json.loads(data)
            logger.info(f"Global tokens loaded for student_id {student_id}.")
            return tokens
        else:
            logger.info(f"No global tokens found for student_id {student_id}.")
            return None
    except Exception as e:
        logger.error(f"Error loading global student tokens from Redis: {str(e)}")
        return None

async def load_global_student_tokens(student_id):
    try:
        return _load_global_student_tokens_sync(student_id)
    except Exception as e:
        logger.error(f"Error in async load_global_student_tokens: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        return None

@with_redis_retry
def _save_student_schedule_sync(student_id, schedule):
    try:
        redis_conn = get_redis_sync()
        redis_conn.set(f"student_schedule:{student_id}", json.dumps(schedule))
        logger.info(f"Schedule cached for student_id {student_id} (no expiration).")
        return True
    except Exception as e:
        logger.error(f"Error saving student schedule to Redis: {str(e)}")
        raise

async def save_student_schedule(student_id, schedule):
    try:
        # Get the result directly without creating a coroutine
        result = _save_student_schedule_sync(student_id, schedule)
        return result
    except Exception as e:
        logger.error(f"Error in async save_student_schedule: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        raise

@with_redis_retry
def _load_student_schedule_sync(student_id):
    try:
        redis_conn = get_redis_sync()
        data = redis_conn.get(f"student_schedule:{student_id}")
        if data:
            schedule = json.loads(data)
            logger.info(f"Schedule loaded from cache for student_id {student_id}.")
            return schedule
        else:
            logger.info(f"No cached schedule found for student_id {student_id}.")
            return None
    except Exception as e:
        logger.error(f"Error loading student schedule from Redis: {str(e)}")
        return None

async def load_student_schedule(student_id):
    try:
        # Get the result directly without creating a coroutine
        result = _load_student_schedule_sync(student_id)
        return result
    except Exception as e:
        logger.error(f"Error in async load_student_schedule: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        return None

def get_basic_auth_header():
    """Generate Basic Auth header from environment variables."""
    if not OAUTH_CLIENT_SECRET:
        logger.warning("OAuth client secret not configured! Token refresh may fail.")
    
    credentials = f"{OAUTH_CLIENT_ID}:{OAUTH_CLIENT_SECRET}"
    encoded = base64.b64encode(credentials.encode()).decode()
    return f"Basic {encoded}"

def refresh_access_token_sync(refresh_token: str) -> dict:
    """Synchronous version to refresh the access token using the refresh token."""
    token_url = "https://sso.bracu.ac.bd/realms/bracu/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": "slm",  # Using the working client_id from old implementation
        "refresh_token": refresh_token,
    }
    
    try:
        import requests
        logger.debug(f"Trying token refresh at: {token_url} (sync)")
        resp = requests.post(token_url, data=data, timeout=10.0)
        logger.debug(f"Token refresh response: {resp.status_code} {resp.text}")
        
        if resp.status_code == 200:
            try:
                new_tokens = resp.json()
                if isinstance(new_tokens, dict) and "access_token" in new_tokens:
                    logger.info("Successfully refreshed access token (sync)")
                    now = int(time.time())
                    
                    # Get expiration from new access token
                    access_jwt_data = decode_jwt_token(new_tokens["access_token"])
                    if "exp" in access_jwt_data:
                        new_tokens["expires_at"] = access_jwt_data["exp"]
                        new_tokens["expires_in"] = max(0, access_jwt_data["exp"] - now)
                    
                    # If we got a new refresh token, get its expiration
                    if "refresh_token" in new_tokens:
                        refresh_jwt_data = decode_jwt_token(new_tokens["refresh_token"])
                        if "exp" in refresh_jwt_data:
                            new_tokens["refresh_expires_at"] = refresh_jwt_data["exp"]
                        else:
                            new_tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes default
                    else:
                        # Keep the old refresh token if we didn't get a new one
                        new_tokens["refresh_token"] = refresh_token
                        refresh_jwt_data = decode_jwt_token(refresh_token)
                        if "exp" in refresh_jwt_data:
                            new_tokens["refresh_expires_at"] = refresh_jwt_data["exp"]
                        else:
                            new_tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes default
                    
                    logger.info(f"New tokens (sync): Access expires in {new_tokens.get('expires_in')}s, "
                              f"Refresh expires in {new_tokens.get('refresh_expires_at', 0) - now}s")
                    return new_tokens
                else:
                    logger.error(f"Invalid token refresh response format (sync)")
                    return None
            except Exception as e:
                logger.error(f"Failed to parse token refresh response (sync): {str(e)}")
                return None
        elif resp.status_code == 401:
            logger.error("Refresh token has expired or is invalid (sync)")
            return None
        else:
            logger.error(f"Failed to refresh token (sync): {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        logger.error(f"Error refreshing token (sync): {str(e)}")
        return None

async def refresh_access_token(refresh_token: str) -> dict:
    """Refresh the access token using the refresh token."""
    token_url = "https://sso.bracu.ac.bd/realms/bracu/protocol/openid-connect/token"
    data = {
        "grant_type": "refresh_token",
        "client_id": "slm",  # Using the working client_id from old implementation
        "refresh_token": refresh_token,
    }
    
    try:
        async with httpx.AsyncClient() as client:
            logger.debug(f"Trying token refresh at: {token_url}")
            resp = await client.post(token_url, data=data, timeout=10.0)
            logger.debug(f"Token refresh response: {resp.status_code} {resp.text}")
            
            if resp.status_code == 200:
                try:
                    new_tokens = resp.json()
                    if isinstance(new_tokens, dict) and "access_token" in new_tokens:
                        logger.info("Successfully refreshed access token")
                        now = int(time.time())
                        
                        # Get expiration from new access token
                        access_jwt_data = decode_jwt_token(new_tokens["access_token"])
                        if "exp" in access_jwt_data:
                            new_tokens["expires_at"] = access_jwt_data["exp"]
                            new_tokens["expires_in"] = max(0, access_jwt_data["exp"] - now)
                        
                        # If we got a new refresh token, get its expiration
                        if "refresh_token" in new_tokens:
                            refresh_jwt_data = decode_jwt_token(new_tokens["refresh_token"])
                            if "exp" in refresh_jwt_data:
                                new_tokens["refresh_expires_at"] = refresh_jwt_data["exp"]
                            else:
                                new_tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes default
                        else:
                            # Keep the old refresh token if we didn't get a new one
                            new_tokens["refresh_token"] = refresh_token
                            refresh_jwt_data = decode_jwt_token(refresh_token)
                            if "exp" in refresh_jwt_data:
                                new_tokens["refresh_expires_at"] = refresh_jwt_data["exp"]
                            else:
                                new_tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes default
                        
                        logger.info(f"New tokens: Access expires in {new_tokens.get('expires_in')}s, "
                                  f"Refresh expires in {new_tokens.get('refresh_expires_at', 0) - now}s")
                        return new_tokens
                    else:
                        logger.error(f"Invalid token refresh response format")
                        return None
                except Exception as e:
                    logger.error(f"Failed to parse token refresh response: {str(e)}")
                    return None
            elif resp.status_code == 401:
                logger.error("Refresh token has expired or is invalid")
                return None
            else:
                logger.error(f"Failed to refresh token: {resp.status_code} {resp.text}")
                return None
    except Exception as e:
        logger.error(f"Error refreshing token: {str(e)}")
        return None

def is_token_expired(tokens, buffer=30):  # Reduced buffer for 5-min access tokens
    """Check if tokens are expired with a buffer time."""
    if not tokens:
        return True
    now = int(time.time())
    # Check access token expiration
    if "expires_at" in tokens:
        if now + buffer >= tokens["expires_at"]:
            logger.info(f"Token expiring soon: expires at {tokens['expires_at']}, current time {now}, buffer {buffer}s")
            return True
    
    # Also check refresh token expiration
    if "refresh_expires_at" in tokens:
        if now >= tokens["refresh_expires_at"]:
            logger.info(f"Refresh token expired: expires at {tokens['refresh_expires_at']}, current time {now}")
            return True
    
    return False

@with_redis_retry
def get_latest_valid_token_sync():
    """Get the most recent valid token from Redis synchronously."""
    try:
        redis_conn = get_redis_sync()
            
        # Get all token keys
        token_keys = redis_conn.keys("tokens:*")
        if not token_keys:
            logger.warning("No tokens found in Redis")
            return None
            
        latest_token = None
        latest_expiry = 0
        session_id = None
        
        # First try to find the most recent valid token
        for key in token_keys:
            tokens_str = redis_conn.get(key)
            if tokens_str:
                try:
                    tokens = json.loads(tokens_str)
                    if "expires_at" in tokens:
                        # If this token expires later than our current latest, update it
                        if tokens["expires_at"] > latest_expiry:
                            latest_token = tokens
                            latest_expiry = tokens["expires_at"]
                            session_id = key.split(":")[-1]
                except json.JSONDecodeError:
                    continue
        
        # If we found a valid token that doesn't need refresh, use it
        if latest_token and not is_token_expired(latest_token):
            logger.info("Using existing valid token")
            return latest_token.get("access_token")
        
        logger.warning("No valid tokens found")
        return None
    except Exception as e:
        logger.error(f"Error in get_latest_valid_token_sync: {str(e)}")
        return None

@with_redis_retry
async def get_latest_valid_token():
    """Get the most recent valid token from Redis, attempting to refresh if needed."""
    try:
        redis_conn = get_redis_sync()
        now = int(time.time())
            
        # Get all token keys
        token_keys = redis_conn.keys("tokens:*")
        if not token_keys:
            logger.warning("No tokens found in Redis")
            return None
            
        latest_token = None
        latest_expiry = 0
        needs_refresh = True
        session_id = None
        
        # First try to find the most recent valid token
        for key in token_keys:
            tokens_str = redis_conn.get(key)
            if tokens_str:
                try:
                    tokens = json.loads(tokens_str)
                    if "expires_at" in tokens:
                        # Check if this token expires later than our current latest
                        if tokens["expires_at"] > latest_expiry:
                            latest_token = tokens
                            latest_expiry = tokens["expires_at"]
                            session_id = key.split(":")[-1]
                            
                            # Check if token needs proactive refresh
                            needs_refresh = is_token_expired(tokens)
                            if not needs_refresh and "refresh_token" in tokens:
                                total_lifetime = tokens["expires_at"] - (tokens.get("activated_at") or (tokens["expires_at"] - 300))
                                time_elapsed = now - (tokens.get("activated_at") or (tokens["expires_at"] - 300))
                                if time_elapsed >= (total_lifetime * 0.5):  # Refresh at 50% for 5-min tokens
                                    needs_refresh = True
                                    debug_print(f"Latest token needs proactive refresh (50% lifetime elapsed)")
                except json.JSONDecodeError:
                    continue
        
        # If we found a valid token that doesn't need refresh, use it
        if latest_token and not needs_refresh:
            logger.info("Using existing valid token")
            return latest_token.get("access_token")
        
        # If we have a token but it needs refresh, try to refresh it
        if latest_token and "refresh_token" in latest_token and session_id:
            logger.info("Attempting to refresh token")
            try:
                new_tokens = refresh_access_token_sync(latest_token["refresh_token"])
                if new_tokens and "access_token" in new_tokens:
                    # Add activation time and save the refreshed tokens
                    new_tokens["activated_at"] = now
                    _save_tokens_to_redis_sync(session_id, new_tokens)
                    logger.info("Successfully refreshed token")
                    return new_tokens.get("access_token")
                else:
                    logger.error("Token refresh failed - invalid response")
                    # Delete the expired/invalid tokens
                    redis_conn.delete(f"tokens:{session_id}")
            except Exception as e:
                logger.error(f"Error refreshing token: {str(e)}")
                # Delete the expired/invalid tokens
                redis_conn.delete(f"tokens:{session_id}")
        
        # Try to refresh other tokens even if the latest one failed
        logger.info("Attempting to refresh other available tokens")
        for key in token_keys:
            try:
                tokens_str = redis_conn.get(key)
                if tokens_str:
                    tokens = json.loads(tokens_str)
                    if "refresh_token" in tokens:
                        session_id = key.split(":")[-1]
                        new_tokens = refresh_access_token_sync(tokens["refresh_token"])
                        if new_tokens and "access_token" in new_tokens:
                            new_tokens["activated_at"] = now
                            _save_tokens_to_redis_sync(session_id, new_tokens)
                            logger.info("Successfully refreshed token from backup session")
                            return new_tokens.get("access_token")
                        else:
                            # Clean up failed refresh tokens
                            redis_conn.delete(f"tokens:{session_id}")
            except Exception as e:
                logger.error(f"Error refreshing backup token: {str(e)}")
                continue
        
        logger.warning("No valid tokens found and refresh attempts failed")
        return None
    except Exception as e:
        logger.error(f"Error in get_latest_valid_token: {str(e)}")
        return None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    session_id = request.session.get("id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        request.session["id"] = session_id
    
    # Calculate network uptime
    network_uptime_seconds = int(time.time() - start_time)
    days, remainder = divmod(network_uptime_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    network_uptime_str = []
    if days:
        network_uptime_str.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        network_uptime_str.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        network_uptime_str.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds or not network_uptime_str:
        network_uptime_str.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    network_uptime_display = 'Network uptime: ' + ', '.join(network_uptime_str)
    
    # Calculate token remaining time
    token_remaining_display = "No active token."
    section_status_display = ""
    global_token_display = ""
    
    # Get global token status
    try:
        redis_conn = get_redis_sync()
        token_keys = redis_conn.keys("tokens:*")
        
        if token_keys:
            total_sessions = len(token_keys)
            active_sessions = 0
            earliest_expiry = None
            earliest_created = None
            
            for key in token_keys:
                try:
                    tokens_str = redis_conn.get(key)
                    if tokens_str:
                        tokens = json.loads(tokens_str)
                        if tokens and "access_token" in tokens and not is_token_expired(tokens):
                            active_sessions += 1
                            if "expires_at" in tokens:
                                expiry = tokens["expires_at"]
                                if earliest_expiry is None or expiry < earliest_expiry:
                                    earliest_expiry = expiry
                            # Get creation time from activated_at field
                            created_at = tokens.get("activated_at", tokens.get("created_at"))
                            if created_at is None:
                                # Fallback to current time if no creation timestamp
                                created_at = int(time.time())
                            if earliest_created is None or created_at < earliest_created:
                                earliest_created = created_at
                except:
                    continue
            
            if active_sessions > 0 and earliest_created:
                now = int(time.time())
                uptime_seconds = max(0, now - earliest_created)
                remaining = max(0, earliest_expiry - now)
                
                # Format uptime
                uptime_parts = []
                if uptime_seconds >= 86400:
                    days = uptime_seconds // 86400
                    uptime_parts.append(f"{days}d")
                    hours = (uptime_seconds % 86400) // 3600
                    if hours > 0:
                        uptime_parts.append(f"{hours}h")
                elif uptime_seconds >= 3600:
                    hours = uptime_seconds // 3600
                    uptime_parts.append(f"{hours}h")
                    minutes = (uptime_seconds % 3600) // 60
                    if minutes > 0:
                        uptime_parts.append(f"{minutes}m")
                else:
                    minutes = uptime_seconds // 60
                    if minutes > 0:
                        uptime_parts.append(f"{minutes}m")
                    seconds = uptime_seconds % 60
                    uptime_parts.append(f"{seconds}s")
                
                # Format remaining time
                remaining_parts = []
                if remaining > 0:
                    days, remainder = divmod(remaining, 86400)
                    hours, remainder = divmod(remainder, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    
                    if days:
                        remaining_parts.append(f"{days}d")
                    if hours:
                        remaining_parts.append(f"{hours}h")
                    if minutes:
                        remaining_parts.append(f"{minutes}m")
                    if seconds or not remaining_parts:
                        remaining_parts.append(f"{seconds}s")
                
                uptime_str = " ".join(uptime_parts)
                remaining_str = " ".join(remaining_parts)
                
                global_token_display = f'<div class="info-box success">🟢 Global token active: {active_sessions}/{total_sessions} sessions valid (uptime: {uptime_str}, expires in {remaining_str})</div>'
            else:
                global_token_display = f'<div class="info-box">🔴 No active global tokens: {total_sessions} expired sessions</div>'
        else:
            global_token_display = '<div class="info-box">🔴 No tokens stored in system</div>'
    except Exception:
        global_token_display = '<div class="info-box error">⚠️ Unable to check global token status</div>'
    
    # Check user's personal token
    try:
        tokens = _load_tokens_from_redis_sync(session_id)
        if tokens and "access_token" in tokens and not is_token_expired(tokens):
            token = tokens["access_token"]
            
            # Check section count if we have a valid token
            with httpx.Client() as client:
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0",
                    "Origin": "https://connect.bracu.ac.bd",
                    "Referer": "https://connect.bracu.ac.bd/"
                }
                
                url = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections"
                resp = client.get(url, headers=headers)
                
                if resp.status_code == 200:
                    sections = resp.json()
                    has_changed, current, stored = check_section_changes_sync(sections)
                    if has_changed:
                        section_status_display = f'<div class="section-status warning">Section count changed from {stored} to {current}. Consider updating lab cache.</div>'
            
            # Calculate token remaining time
            now = int(time.time())
            if "expires_at" in tokens:
                remaining = max(0, tokens["expires_at"] - now)
                days, remainder = divmod(remaining, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                remaining_str = []
                if days:
                    remaining_str.append(f"{days} day{'s' if days != 1 else ''}")
                if hours:
                    remaining_str.append(f"{hours} hour{'s' if hours != 1 else ''}")
                if minutes:
                    remaining_str.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
                if seconds or not remaining_str:
                    remaining_str.append(f"{seconds} second{'s' if seconds != 1 else ''}")
                token_remaining_display = 'Current token active for: ' + ', '.join(remaining_str)
    except Exception:
        pass

    html_content = f"""
    <html><head><title>BRACU Schedule Viewer</title>
    <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }}
    .container {{ max-width: 480px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 40px 32px; }}
    h1 {{ color: #2d3748; margin-bottom: 12px; }}
    .desc {{ color: #4a5568; margin-bottom: 24px; }}
    .button-container {{ display: flex; gap: 12px; justify-content: center; margin-bottom: 24px; flex-wrap: wrap; }}
    .button {{ background: #3182ce; color: #fff; border: none; border-radius: 6px; padding: 10px 22px; font-size: 1rem; cursor: pointer; text-decoration: none; transition: background 0.2s; }}
    .button:hover {{ background: #225ea8; }}
    .button.update {{ background: #38a169; }}
    .button.update:hover {{ background: #2f855a; }}
    .session-id {{ font-size: 0.9em; color: #718096; margin-top: 18px; text-align: center; }}
    .network-uptime {{ font-size: 0.9em; color: #718096; margin-top: 8px; text-align: center; }}
    .token-remaining {{ font-size: 0.9em; color: #718096; margin-top: 4px; text-align: center; }}
    .section-status {{ font-size: 0.9em; margin: 12px 0; padding: 10px; border-radius: 6px; text-align: center; }}
    .section-status.warning {{ background: #fef3c7; color: #92400e; }}
    .footer {{ font-size: 0.9em; color: #a0aec0; margin-top: 32px; text-align: center; }}
    .footer a {{ color: #3182ce; text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}
    #updateStatus {{ display: none; margin-top: 12px; padding: 10px; border-radius: 6px; background: #e9ecef; }}
    .progress {{ width: 100%; height: 4px; background: #e2e8f0; border-radius: 2px; margin-top: 8px; }}
    .progress-bar {{ width: 0%; height: 100%; background: #3182ce; border-radius: 2px; transition: width 0.3s ease; }}
    .info-box {{ background: #ebf8ff; border-radius: 6px; padding: 12px; margin: 12px 0; color: #2c5282; font-size: 0.9em; }}
    .info-box.error {{ background: #fed7d7; color: #c53030; }}
    .info-box.warning {{ background: #fef3c7; color: #92400e; }}
    .info-box.success {{ background: #c6f6d5; color: #2f855a; }}
    button {{ background: #3182ce; color: #fff; border: none; border-radius: 6px; padding: 8px 16px; font-size: 0.9rem; cursor: pointer; transition: background 0.2s; }}
    button:hover {{ background: #225ea8; }}
    </style>
    <script>
    let currentTaskId = null;
    let statusCheckInterval = null;
    let tokenCheckInterval = null;
    let isPageVisible = true;
    let retryCount = 0;
    const MAX_RETRIES = 3;

    // Handle page visibility changes
    document.addEventListener('visibilitychange', function() {{
        isPageVisible = !document.hidden;
        if (isPageVisible && currentTaskId) {{
            checkStatus(); // Check immediately when page becomes visible
        }}
    }});

    async function updateLabs() {{
        try {{
            const response = await fetch('/update-labs', {{ method: 'POST' }});
            const data = await response.json();
            
            if (response.ok) {{
                currentTaskId = data.task_id;
                retryCount = 0; // Reset retry count
                const statusEl = document.getElementById('updateStatus');
                statusEl.style.display = 'block';
                statusEl.innerHTML = `
                    <div>Update started! You can close this tab - the update will continue in the background.</div>
                    <div class="info-box">
                        <strong>Note:</strong> This process may take a few minutes. You can return to this page anytime to check the status.
                    </div>
                    <div class="progress">
                        <div class="progress-bar" style="width: 0%"></div>
                    </div>
                `;
                startStatusCheck();
            }} else {{
                showError('Error: ' + (data.error || 'Failed to start update'));
            }}
        }} catch (error) {{
            showError('Error: ' + error.message);
        }}
    }}

    function showError(message) {{
        const statusEl = document.getElementById('updateStatus');
        statusEl.style.display = 'block';
        statusEl.innerHTML = `
            <div class="info-box error">
                ${{message}}
                <br><br>
                <button onclick="location.reload()">Refresh Page</button>
            </div>
        `;
    }}

    async function checkStatus() {{
        if (!currentTaskId) return;
        
        try {{
            const response = await fetch(`/update-labs/status/${{currentTaskId}}`);
            const data = await response.json();
            
            const statusEl = document.getElementById('updateStatus');
            
            if (response.ok) {{
                retryCount = 0; // Reset retry count on successful response
                
                if (data.status === 'completed' || data.status === 'error' || data.status === 'cancelled') {{
                    clearInterval(statusCheckInterval);
                    let statusClass = data.status === 'completed' ? 'success' : 'error';
                    let cleanupMessage = '';
                    if (data.cleanup_in !== undefined) {{
                        cleanupMessage = `<br><small>(Status will be available for ${{Math.ceil(data.cleanup_in / 60)}} minutes)</small>`;
                    }}
                    statusEl.innerHTML = `
                        <div class="info-box ${{statusClass}}">
                            ${{data.message}}
                            ${{cleanupMessage}}
                            ${{data.status === 'completed' ? '<br>Page will refresh in 3 seconds...' : ''}}
                        </div>
                    `;
                    if (data.status === 'completed' && isPageVisible) {{
                        setTimeout(() => {{
                            statusEl.style.display = 'none';
                            location.reload();
                        }}, 3000);
                    }}
                }} else {{
                    statusEl.innerHTML = `
                        <div>${{data.message}}</div>
                        <div class="info-box">
                            <strong>Note:</strong> This process may take a few minutes. You can close this tab - the update will continue in the background.
                        </div>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${{data.progress}}%"></div>
                        </div>
                    `;
                }}
            }} else if (response.status === 404) {{
                // Task not found or expired
                clearInterval(statusCheckInterval);
                statusEl.innerHTML = `
                    <div class="info-box warning">
                        ${{data.message}}
                        <br><br>
                        <button onclick="location.reload()">Refresh Page</button>
                    </div>
                `;
            }} else {{
                retryCount++;
                if (retryCount >= MAX_RETRIES) {{
                    clearInterval(statusCheckInterval);
                    showError('Failed to check status after multiple attempts. Please refresh the page.');
                }}
            }}
        }} catch (error) {{
            console.error('Error checking status:', error);
            retryCount++;
            if (retryCount >= MAX_RETRIES) {{
                clearInterval(statusCheckInterval);
                showError('Failed to check status after multiple attempts. Please refresh the page.');
            }}
        }}
    }}

    function startStatusCheck() {{
        if (statusCheckInterval) clearInterval(statusCheckInterval);
        statusCheckInterval = setInterval(checkStatus, 1000);
        checkStatus();  // Check immediately
    }}

    async function checkTokenStatus() {{
        try {{
            const response = await fetch('/token-status');
            const data = await response.json();
            
            const tokenStatusEl = document.getElementById('token-status-display');
            if (!tokenStatusEl) return;
            
            let statusClass = '';
            let statusText = '';
            
            if (data.valid) {{
                statusClass = 'success';
                statusText = `Token Valid ✓ (ID: ${{data.id}})`;
            }} else {{
                statusClass = 'error';
                statusText = 'Token Expired ✗';
            }}
            
            tokenStatusEl.innerHTML = `<div class="info-box ${{statusClass}}">${{statusText}}</div>`;
            
        }} catch (error) {{
            console.error('Error checking token status:', error);
        }}
    }}

    function startTokenCheck() {{
        if (tokenCheckInterval) clearInterval(tokenCheckInterval);
        tokenCheckInterval = setInterval(checkTokenStatus, 5000); // Check every 5 seconds
        checkTokenStatus(); // Check immediately
    }}

    // Check for existing task on page load
    window.onload = function() {{
        const urlParams = new URLSearchParams(window.location.search);
        const taskId = urlParams.get('task_id');
        if (taskId) {{
            currentTaskId = taskId;
            document.getElementById('updateStatus').style.display = 'block';
            startStatusCheck();
        }}
        startTokenCheck(); // Start token validation on page load
    }};
    </script></head><body>
    <div class='container'>
        <h1>BRACU Schedule Viewer</h1>
        <div class='desc'>A simple client to view your BRACU Connect schedule.<br>Session-based, no password required.</div>
        <div class='button-container'>
            <a class='button' href='/enter-tokens'>Enter Tokens</a>
            <a class='button' href='/mytokens'>View Tokens</a>
            <a class='button' href='/raw-schedule'>View Raw Schedule</a>
            <button class='button update' onclick='updateLabs()'>Update Lab Cache</button>
        </div>
        {section_status_display}
        <div id='updateStatus'></div>
        <div class='session-id'>Session: {session_id}</div>
        <div class='network-uptime'>{network_uptime_display}</div>
        <div class='token-remaining'>{token_remaining_display}</div>
        {global_token_display}
        <div id='token-status-display' style='margin-top: 8px; text-align: center;'></div>
        <div class='footer'>API server by <b>Wasif Faisal</b> to support <a href='https://routinez.vercel.app/' target='_blank'>Routinez</a></div>
    </div></body></html>
    """
    return HTMLResponse(html_content)

@app.get("/enter-tokens", response_class=HTMLResponse)
async def enter_tokens_form(request: Request):
    session_id = request.session.get("id")
    if not session_id:
        # No session, redirect to home
        return RedirectResponse("/", status_code=302)
    html_content = """
    <html><head><title>Enter Tokens</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }
    .container { max-width: 420px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; }
    h2 { color: #2d3748; margin-bottom: 18px; }
    form { display: flex; flex-direction: column; gap: 16px; }
    input { padding: 10px; border-radius: 6px; border: 1px solid #cbd5e0; font-size: 1rem; }
    button { background: #3182ce; color: #fff; border: none; border-radius: 6px; padding: 10px 0; font-size: 1rem; cursor: pointer; transition: background 0.2s; }
    button:hover { background: #225ea8; }
    .back { display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }
    .back:hover { text-decoration: underline; }
    </style></head><body>
    <div class='container'>
        <h2>Enter Your Tokens</h2>
        <form action='/enter-tokens' method='post'>
            <input name='access_token' placeholder='Access Token' required autocomplete='off'>
            <input name='refresh_token' placeholder='Refresh Token' required autocomplete='off'>
            <button type='submit'>Save Tokens</button>
        </form>
        <a class='back' href='/'>Back to Home</a>
    </div></body></html>
    """
    return HTMLResponse(html_content)

@app.post("/enter-tokens", response_class=HTMLResponse)
async def save_tokens_form(request: Request, access_token: str = Form(...), refresh_token: str = Form(...)):
    session_id = request.session.get("id")
    if not session_id:
        # No session, redirect to home
        return RedirectResponse("/", status_code=302)
    
    # Get token expiration from JWT
    now = int(time.time())
    access_jwt_data = decode_jwt_token(access_token)
    refresh_jwt_data = decode_jwt_token(refresh_token)
    
    tokens = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": access_jwt_data.get("exp", now + 300),  # 5 minutes default
        "refresh_expires_at": refresh_jwt_data.get("exp", now + 1800)  # 30 minutes default
    }
    # Set activated_at if not already present in Redis
    existing = await load_tokens_from_redis(session_id)
    if existing and "activated_at" in existing:
        tokens["activated_at"] = existing["activated_at"]
    else:
        tokens["activated_at"] = now
    
    await save_tokens_to_redis(session_id, tokens)
    html_content = """
    <html><head><title>Tokens Saved</title>
    <style>
    body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }
    .container { max-width: 420px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; text-align: center; }
    .msg { color: #2d3748; font-size: 1.1em; margin-bottom: 18px; }
    .back { display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }
    .back:hover { text-decoration: underline; }
    </style></head><body>
    <div class='container'>
        <div class='msg'>Tokens saved successfully!</div>
        <a class='back' href='/'>Back to Home</a>
    </div></body></html>
    """
    return HTMLResponse(html_content)

@app.get("/mytokens", response_class=HTMLResponse)
async def view_tokens(request: Request):
    """View tokens for the current authenticated session."""
    try:

        current_session = request.session.get("id")
        if not current_session:
            # No session, redirect to home with security message
            return HTMLResponse("""
                <html><head><title>Access Denied</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }
                .container { max-width: 520px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; text-align: center; }
                .error { color: #e53e3e; margin-bottom: 18px; font-weight: 600; }
                .back { display: block; margin-top: 18px; color: #3182ce; text-decoration: none; font-weight: 500; }
                .back:hover { text-decoration: underline; }
                </style></head><body>
                <div class='container'>
                    <div class='error'>🔒 Access Denied</div>
                    <p>You must access this page from the main application.</p>
                    <a class='back' href='/'>← Back to Home</a>
                </div></body></html>
            """, status_code=403)

        # Load tokens for the current session first
        redis_conn = get_redis_sync()
        tokens = _load_tokens_from_redis_sync(current_session)
        
        # If no session tokens, try to get global tokens
        global_tokens = None
        global_session = None
        if not tokens or not tokens.get("access_token"):
            try:
                all_keys = redis_conn.keys("tokens:*")
                for key in all_keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                    session_id = key_str.replace("tokens:", "")
                    if session_id and session_id != current_session:
                        global_tokens = _load_tokens_from_redis_sync(session_id)
                        if global_tokens and global_tokens.get("access_token"):
                            global_session = session_id
                            break
            except Exception as e:
                logger.error(f"Error loading global tokens: {str(e)}")
        
        # Use tokens (session or global) for display
        display_tokens = tokens if (tokens and tokens.get("access_token")) else global_tokens
        display_session = current_session if (tokens and tokens.get("access_token")) else global_session
        
        # Calculate token expiration times if tokens exist
        token_info = ""
        if display_tokens:
            now = int(time.time())
            access_expires_in = max(0, display_tokens.get("expires_at", 0) - now)
            refresh_expires_in = max(0, display_tokens.get("refresh_expires_at", 0) - now)
            
            token_info = f"""
            <div class='token-info'>
                <div class='expiry'>Access token expires in: {access_expires_in} seconds</div>
                <div class='expiry'>Refresh token expires in: {refresh_expires_in} seconds</div>
            </div>
            """

        token_source = "Session" if (tokens and tokens.get("access_token")) else ("Global" if (global_tokens and global_tokens.get("access_token")) else None)
        
        html_content = f"""
        <html><head><title>My Tokens</title>
        <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }}
        .container {{ max-width: 520px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; }}
        h2 {{ color: #2d3748; margin-bottom: 18px; }}
        pre {{ background: #f7fafc; border-radius: 6px; padding: 18px; font-size: 1em; color: #2d3748; overflow-x: auto; }}
        .msg {{ color: #e53e3e; margin-bottom: 18px; }}
        .info {{ color: #2b6cb0; margin-bottom: 18px; }}
        .back {{ display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }}
        .back:hover {{ text-decoration: underline; }}
        .token-info {{ margin: 12px 0; padding: 12px; background: #ebf8ff; border-radius: 6px; }}
        .expiry {{ color: #2b6cb0; margin: 4px 0; }}
        .session {{ font-size: 0.9em; color: #718096; margin-top: 12px; }}
        .token-source {{ font-size: 0.9em; color: #4a5568; margin-bottom: 12px; font-weight: bold; }}
        </style></head><body>
        <div class='container'>
            <h2>Your Tokens</h2>
            {f'<div class="token-source">Token Source: {token_source}</div>' if token_source else '<div class="msg">No tokens found</div>'}
            {token_info if display_tokens else ''}
            {('<pre>' + json.dumps(display_tokens, indent=2) + '</pre>') if display_tokens else ''}
            <div class='session'>Session ID: {display_session}</div>
            <a class='back' href='/'>Back to Home</a>
        </div></body></html>
        """
        return HTMLResponse(html_content)
    except Exception as e:
        logger.error(f"Error in view_tokens: {str(e)}")
        return HTMLResponse(f"""
            <html><head><title>Error</title>
            <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }}
            .container {{ max-width: 520px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; text-align: center; }}
            .error {{ color: #e53e3e; margin-bottom: 18px; }}
            .back {{ display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }}
            .back:hover {{ text-decoration: underline; }}
            </style></head><body>
            <div class='container'>
                <div class='error'>An error occurred while loading tokens.</div>
                <div class='error'>{str(e)}</div>
                <a class='back' href='/'>Back to Home</a>
            </div></body></html>
        """, status_code=500)

async def get_seat_status(token: str) -> dict:
    """Get real-time seat status for sections in student's schedule."""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://connect.bracu.ac.bd",
            "Referer": "https://connect.bracu.ac.bd/"
        }
        
        async with httpx.AsyncClient() as client:
            # Get the seat status for all sections
            url = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections/seat-status"
            resp = await client.get(url, headers=headers)
            
            if resp.status_code == 200:
                return resp.json()  # This returns a dict of section_id -> booked_seats
            else:
                logger.error(f"Failed to get seat status: {resp.status_code} {resp.text}")
                return {}
    except Exception as e:
        logger.error(f"Error getting seat status: {str(e)}")
        return {}

# Add these functions before the raw_schedule endpoint
async def get_section_count_from_redis():
    """Get the stored section count from Redis."""
    try:
        redis_conn = get_redis_sync()
        count = redis_conn.get("total_section_count")
        return int(count) if count else None
    except Exception as e:
        logger.error(f"Error getting section count from Redis: {str(e)}")
        return None

async def save_section_count_to_redis(count: int):
    """Save the current section count to Redis."""
    try:
        redis_conn = get_redis_sync()
        redis_conn.set("total_section_count", str(count))
        logger.info(f"Saved section count to Redis: {count}")
    except Exception as e:
        logger.error(f"Error saving section count to Redis: {str(e)}")

# Initialize empty lab cache
lab_cache = {}

def save_lab_cache_to_redis_sync(lab_data: dict):
    """Save lab section data to Redis synchronously."""
    try:
        redis_conn = get_redis_sync()
        redis_conn.set("lab_cache", json.dumps(lab_data))
        logger.info(f"Saved {len(lab_data)} lab sections to Redis cache")
        return True
    except Exception as e:
        logger.error(f"Error saving lab cache to Redis: {str(e)}")
        return False

async def save_lab_cache_to_redis(lab_data: dict):
    """Save lab section data to Redis."""
    return save_lab_cache_to_redis_sync(lab_data)

def load_lab_cache_from_redis_sync():
    """Load lab section data from Redis synchronously."""
    try:
        redis_conn = get_redis_sync()
        data = redis_conn.get("lab_cache")
        if data:
            return json.loads(data)
        return {}
    except Exception as e:
        logger.error(f"Error loading lab cache from Redis: {str(e)}")
        return {}

async def load_lab_cache_from_redis():
    """Load lab section data from Redis."""
    return load_lab_cache_from_redis_sync()

def deduplicate_lab_cache_sync():
    """Deduplicate lab cache based on labSectionId to prevent duplicates synchronously."""
    global lab_cache
    seen_lab_sections = set()
    seen_parent_sections = set()
    deduplicated_cache = {}
    duplicates_removed = 0
    
    for parent_section_id, lab_info in lab_cache.items():
        lab_section_id = str(lab_info.get("labSectionId", ""))
        
        # Skip if parent_section_id is duplicated
        if parent_section_id in seen_parent_sections:
            duplicates_removed += 1
            continue
            
        # Check for duplicate labSectionId
        if lab_section_id and lab_section_id in seen_lab_sections:
            duplicates_removed += 1
            continue
            
        # Add to seen sets
        seen_parent_sections.add(parent_section_id)
        if lab_section_id:
            seen_lab_sections.add(lab_section_id)
            
        deduplicated_cache[parent_section_id] = lab_info
    
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate lab sections from cache")
        lab_cache = deduplicated_cache
        save_lab_cache_to_redis_sync(lab_cache)
    
    return len(lab_cache)

async def deduplicate_lab_cache():
    """Deduplicate lab cache based on labSectionId to prevent duplicates."""
    return deduplicate_lab_cache_sync()

def validate_lab_cache_sync():
    """Validate lab cache data integrity and remove invalid entries synchronously."""
    global lab_cache
    invalid_entries = 0
    valid_cache = {}
    
    for parent_section_id, lab_info in lab_cache.items():
        # Check if lab_info has required fields
        if not isinstance(lab_info, dict):
            invalid_entries += 1
            continue
            
        lab_section_id = lab_info.get("labSectionId")
        lab_course_code = lab_info.get("labCourseCode")
        
        # Validate required fields
        if not lab_section_id or not lab_course_code:
            invalid_entries += 1
            continue
            
        # Validate data types
        if not isinstance(lab_section_id, (str, int)):
            invalid_entries += 1
            continue
            
        valid_cache[parent_section_id] = lab_info
    
    if invalid_entries > 0:
        logger.warning(f"Removed {invalid_entries} invalid lab cache entries")
        lab_cache = valid_cache
        save_lab_cache_to_redis_sync(lab_cache)
    
    return len(lab_cache)

async def validate_lab_cache():
    """Validate lab cache data integrity and remove invalid entries."""
    return validate_lab_cache_sync()

def initialize_lab_cache_sync():
    """Initialize lab cache from Redis synchronously."""
    global lab_cache
    try:
        lab_cache = load_lab_cache_from_redis_sync()
        logger.info(f"Loaded {len(lab_cache)} lab sections from Redis cache")
        
        # Deduplicate cache on initialization
        deduplicate_lab_cache_sync()
        
        # Validate cache integrity
        validate_lab_cache_sync()
        
    except Exception as e:
        logger.warning(f"Could not load lab cache from Redis: {str(e)}. Starting with empty lab cache.")

async def initialize_lab_cache():
    """Initialize lab cache from Redis."""
    return initialize_lab_cache_sync()

# Initialize lab cache on startup
@app.on_event("startup")
def startup_event():
    initialize_lab_cache_sync()

@app.get("/clear-lab-cache")
def clear_lab_cache():
    """Clear the lab cache and reset it to empty."""
    global lab_cache
    try:
        lab_cache = {}
        redis_conn = get_redis_sync()
        redis_conn.delete("lab_cache")
        logger.info("Lab cache cleared successfully")
        return {"message": "Lab cache cleared successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error clearing lab cache: {str(e)}")
        return {"message": f"Error clearing lab cache: {str(e)}", "status": "error"}

def update_lab_cache_sync(token: str, sections: list):
    """Update lab cache with new lab sections found in the provided sections list."""
    global lab_cache
    updated_lab_data = []
    
    # Ensure lab_cache is loaded from Redis if it's empty
    if not lab_cache:
        lab_cache = load_lab_cache_from_redis_sync() or {}
        logger.info(f"Loaded {len(lab_cache)} lab sections from Redis cache")
    
    # Create a lookup dictionary for faster section access
    section_dict = {str(section.get("sectionId")): idx for idx, section in enumerate(sections) if section.get("sectionId")}
    
    # Filter sections that need to be checked for labs - only from the provided sections list
    sections_to_check = [
        section for section in sections 
        if section.get("sectionId") 
        and str(section.get("sectionId")) not in lab_cache
        and section.get("sectionType") != "LAB"  # Skip sections that are already labs
        and (section.get("sectionType") == "THEORY" or section.get("sectionType") == "OTHER")  # Check both THEORY and OTHER sections
    ]
    
    if sections_to_check:
        logger.info(f"Found {len(sections_to_check)} new sections to check for labs in current schedule")
        
        # Process sections in batches to limit concurrent requests
        batch_size = 5  # Process 5 sections at a time
        
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://connect.bracu.ac.bd",
            "Referer": "https://connect.bracu.ac.bd/"
        }
        
        # Process sections in batches using synchronous client
        with httpx.Client(timeout=15.0) as client:
            for i in range(0, len(sections_to_check), batch_size):
                batch = sections_to_check[i:i+batch_size]
                
                for section in batch:
                    section_id = str(section.get("sectionId"))
                    # Skip if already in cache
                    if section_id in lab_cache:
                        continue
                        
                    url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/{section_id}/details"
                    
                    try:
                        response = client.get(url, headers=headers)
                        
                        if response.status_code == 200:
                            details_data = response.json()
                            child_section = details_data.get("childSection")
                            if child_section:
                                # Parse sectionSchedule if it's a string
                                lab_schedule = child_section.get("sectionSchedule")
                                if isinstance(lab_schedule, str):
                                    try:
                                        lab_schedule = json.loads(lab_schedule)
                                    except:
                                        lab_schedule = None
                                
                                # Ensure we properly identify lab sections
                                course_code = child_section.get("courseCode", "")
                                section_name = child_section.get("sectionName", "")
                                
                                # Check if this is a lab section by course code suffix or name
                                is_lab = (
                                    course_code.upper().endswith("L") or
                                    "LAB" in section_name.upper() or
                                    "LABORATORY" in section_name.upper()
                                )
                                
                                lab_info = {
                                    "sectionId": section_id,
                                    "labSectionId": child_section.get("sectionId"),
                                    "labCourseCode": course_code,
                                    "labFaculties": child_section.get("faculties"),
                                    "labName": section_name,
                                    "labRoomName": child_section.get("roomName"),
                                    "labSchedules": lab_schedule,
                                    "isLab": is_lab
                                }
                                
                                # Log MAT120 lab detection
                                if "MAT120" in course_code.upper():
                                    logger.info(f"Found MAT120 lab section: {course_code} - {section_name} (isLab: {is_lab})")
                                
                                updated_lab_data.append(lab_info)
                                
                                # Update the section in the original list using the lookup dictionary
                                if section_id in section_dict:
                                    sections[section_dict[section_id]].update(lab_info)
                                
                                # Update the lab cache
                                lab_cache[section_id] = lab_info
                    except Exception as e:
                        logger.error(f"Error processing section details for section {section_id}: {str(e)}")
                        continue
    
        # Update Redis with any new lab sections found and deduplicate
        if updated_lab_data:
            try:
                # Deduplicate cache before saving
                final_count = deduplicate_lab_cache_sync()
                logger.info(f"Lab cache deduplicated to {final_count} unique lab sections")
                
                # Save updated lab cache to Redis
                save_lab_cache_to_redis_sync(lab_cache)
                logger.info(f"Updated Redis lab cache with {len(updated_lab_data)} new lab sections")
            except Exception as e:
                logger.error(f"Error updating Redis lab cache: {str(e)}")
    else:
        logger.info("No new sections to check for labs in current schedule")
    
    # Apply cached lab data to all sections in the list using the lookup dictionary
    for section_id, lab_data in lab_cache.items():
        if section_id in section_dict:
            sections[section_dict[section_id]].update(lab_data)
    
    return sections

async def update_lab_cache(token: str, sections: list):
    """Async wrapper for update_lab_cache_sync."""
    return update_lab_cache_sync(token, sections)

@asynccontextmanager
async def get_schedule_lock():
    """Get a lock for schedule access that's bound to the current event loop."""
    loop = asyncio.get_running_loop()
    if loop not in schedule_locks:
        schedule_locks[loop] = Lock()
    try:
        async with schedule_locks[loop]:
            yield
    finally:
        if not schedule_locks[loop].locked() and loop in schedule_locks:
            del schedule_locks[loop]

# Removed caching for raw schedule to ensure real-time data

@app.get("/raw-schedule", response_class=JSONResponse)
def raw_schedule(request: Request):
    """Get the raw schedule data. Always attempts to fetch fresh data, only uses cached as last resort."""
    try:
        session_id = request.session.get("id")
        
        # First, try to get a valid token with aggressive refresh
        token = None
        attempts = 0
        max_attempts = 3
        
        while attempts < max_attempts and not token:
            attempts += 1
            if session_id:
                tokens = _load_tokens_from_redis_sync(session_id)
                if tokens and "access_token" in tokens and not is_token_expired(tokens):
                    token = tokens["access_token"]
                    break
                else:
                    # Try to get global token with refresh
                    token = asyncio.run(get_latest_valid_token())
            else:
                token = asyncio.run(get_latest_valid_token())
            
            if not token and attempts < max_attempts:
                logger.info(f"Token refresh attempt {attempts}/{max_attempts} failed, retrying...")
                time.sleep(1)  # Brief delay between attempts

        redis_conn = get_redis_sync()

        if not token:
            # Only use cached data as absolute last resort
            keys = redis_conn.keys("student_schedule:*")
            if keys:
                latest_key = sorted(keys)[-1]
                cached_schedule = redis_conn.get(latest_key)
                if cached_schedule:
                    schedule_data = json.loads(cached_schedule)
                    # Ensure sectionSchedule is parsed from string if needed
                    for section in schedule_data:
                        for schedule_field in ["sectionSchedule", "labSchedules"]:
                            if isinstance(section.get(schedule_field), str):
                                try:
                                    section[schedule_field] = json.loads(section[schedule_field])
                                except:
                                    section[schedule_field] = None
                    
                    response_data = {
                        "cached": True,
                        "data": schedule_data,
                        "warning": "⚠️ No valid tokens available - serving cached data. This data may be outdated. Please add new tokens at /enter-tokens to get fresh schedule data."
                    }
                    
                    return JSONResponse(response_data)
            return JSONResponse({"error": "No valid tokens available. Please add tokens via /enter-tokens endpoint"}, status_code=401)

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://connect.bracu.ac.bd",
            "Referer": "https://connect.bracu.ac.bd/"
        }
        
        # Get student ID and schedule using synchronous client
        with httpx.Client(timeout=15.0) as client:
            # Get student ID
            portfolios_url = "https://connect.bracu.ac.bd/api/mds/v1/portfolios"
            
            resp = client.get(portfolios_url, headers=headers)
            
            if resp.status_code == 401:
                logger.warning("Token unauthorized")
                if session_id:
                    redis_conn.delete(f"tokens:{session_id}")
                return JSONResponse({"error": "Token expired, please refresh page"}, status_code=401)
            
            if resp.status_code != 200:
                return JSONResponse({
                    "error": "Failed to fetch student info", 
                    "status_code": resp.status_code
                }, status_code=resp.status_code)
            
            data = resp.json()
            if not isinstance(data, list) or not data or "id" not in data[0]:
                return JSONResponse({"error": "Could not find student id in response."}, status_code=500)
            
            student_id = data[0]["id"]
            logger.info(f"Got student ID: {student_id}")

            # Check if we have a cached schedule for this student
            cached_schedule = _load_student_schedule_sync(student_id)
            if cached_schedule:
                # Use cached schedule as a fallback if API call fails
                fallback_schedule = cached_schedule
            else:
                fallback_schedule = None

            # Get schedule and seat status sequentially
            schedule_url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/student/{student_id}/schedules"
            seat_status_url = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections/seat-status"
            
            schedule_resp = client.get(schedule_url, headers=headers)
            seat_status_resp = client.get(seat_status_url, headers=headers)
            
            # Handle schedule response
            if schedule_resp.status_code != 200:
                if schedule_resp.status_code == 401:
                    logger.warning("Token unauthorized")
                    if session_id:
                        redis_conn.delete(f"tokens:{session_id}")
                    return JSONResponse({"error": "Token expired, please refresh page"}, status_code=401)
                
                if fallback_schedule:
                    response_data = {
                        "cached": True,
                        "data": fallback_schedule,
                        "warning": "Failed to fetch fresh schedule. Using cached data which may be outdated."
                    }
                    return JSONResponse(response_data)
                return JSONResponse({
                    "error": "Failed to fetch schedule",
                    "status_code": schedule_resp.status_code
                }, status_code=schedule_resp.status_code)
            
            schedule_data = schedule_resp.json()
            
            # Process seat status response
            booked_seats = {}
            if seat_status_resp.status_code == 200:
                try:
                    booked_seats = seat_status_resp.json()
                except Exception as e:
                    logger.error(f"Error parsing seat status: {str(e)}")
            
            # Ensure lab_cache is loaded
            global lab_cache
            if not lab_cache:
                lab_cache = load_lab_cache_from_redis_sync() or {}
                logger.info(f"Loaded {len(lab_cache)} lab sections from Redis cache")
            
            # Create a lookup dictionary for faster section access
            section_dict = {str(section.get("sectionId")): idx for idx, section in enumerate(schedule_data) if section.get("sectionId")}
            
            # Apply cached lab data to schedule sections using the lookup dictionary
            for section_id, lab_data in lab_cache.items():
                if section_id in section_dict:
                    schedule_data[section_dict[section_id]].update(lab_data)
            
            # Update seat status and parse schedules
            for section in schedule_data:
                section_id = str(section.get("sectionId"))
                total_capacity = section.get("capacity", 0)
                current_booked = booked_seats.get(section_id, section.get("consumedSeat", 0))
                
                section["consumedSeat"] = current_booked
                section["realTimeSeatCount"] = max(0, total_capacity - current_booked)
                
                # Parse schedules and ensure they are JSON objects before caching
                for schedule_field in ["sectionSchedule", "labSchedules"]:
                    schedule_value = section.get(schedule_field)
                    if isinstance(schedule_value, str):
                        try:
                            section[schedule_field] = json.loads(schedule_value)
                        except:
                            section[schedule_field] = None
                    elif schedule_value is None:
                        section[schedule_field] = None
            
            # Cache the schedule with parsed JSON objects
            _save_student_schedule_sync(student_id, schedule_data)
            
            response_data = {
                "cached": False,
                "data": schedule_data
            }
            
            return JSONResponse(response_data)
                
    except Exception as e:
        logger.error(f"Error in raw_schedule: {str(e)}")
        return JSONResponse({
            "error": f"Internal server error: {str(e)}"
        }, status_code=500)

# Add global exception handler with better error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_id = secrets.token_hex(4)  # Generate a unique error ID
    error_msg = f"Error ID: {error_id} - {str(exc)}"
    
    # Log the full error details
    logger.error(f"Unhandled exception - {error_msg}")
    if DEBUG_MODE or TRACE_MODE:
        logger.error(traceback.format_exc())
        trace_print(f"Full stack trace for error {error_id}:\n{traceback.format_exc()}")
    
    error_response = ErrorResponse(
        error="Internal server error",
        error_code=f"UNHANDLED_ERROR_{error_id}",
        details={
            "message": str(exc),
            "stack_trace": traceback.format_exc() if DEBUG_MODE else None,
            "error_id": error_id
        } if DEBUG_MODE or TRACE_MODE else {"error_id": error_id}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )

def check_section_changes_sync(sections: list) -> Tuple[bool, int, int]:
    """Check if the total number of sections has changed synchronously."""
    try:
        current_count = len(sections)
        stored_count = get_section_count_from_redis_sync() or 0
        
        # Only update stored count if we have sections
        if current_count > 0:
            has_changed = current_count != stored_count
            if has_changed:
                logger.info(f"Section count changed: {stored_count} -> {current_count}")
                save_section_count_to_redis_sync(current_count)
            return has_changed, current_count, stored_count
        return False, stored_count, stored_count
    except Exception as e:
        logger.error(f"Error checking section changes: {str(e)}")
        return False, 0, 0

async def check_section_changes(sections: list) -> Tuple[bool, int, int]:
    """Async wrapper for check_section_changes_sync."""
    return check_section_changes_sync(sections)

@asynccontextmanager
async def get_task_lock(task_id: str):
    """Get a lock for a specific task."""
    if task_id not in TASK_LOCKS:
        TASK_LOCKS[task_id] = Lock()
    try:
        async with TASK_LOCKS[task_id]:
            yield
    finally:
        if task_id in TASK_LOCKS:
            del TASK_LOCKS[task_id]

async def cleanup_old_tasks():
    """Clean up old completed tasks."""
    now = int(time.time())
    for task_id, task_info in list(BACKGROUND_TASKS.items()):
        if task_info.get("status") in ["completed", "error", "cancelled"]:
            # Keep tasks around for TASK_RETENTION_TIME after completion
            if task_info.get("completion_time", 0) + TASK_RETENTION_TIME < now:
                del BACKGROUND_TASKS[task_id]
                logger.debug(f"Cleaned up old task {task_id}")

async def cleanup_task(task_id: str):
    """Clean up task resources safely."""
    try:
        # Remove from active tasks
        task = asyncio.current_task()
        if task:
            ACTIVE_TASKS.discard(task)
        
        # Clean up task resources
        if task_id in TASK_LOCKS:
            del TASK_LOCKS[task_id]
        
        # Ensure task status is set
        if task_id in BACKGROUND_TASKS:
            if BACKGROUND_TASKS[task_id].get("status") not in ["completed", "error", "cancelled"]:
                BACKGROUND_TASKS[task_id] = {
                    "status": "error",
                    "message": "Task terminated unexpectedly",
                    "completion_time": int(time.time())
                }
    except Exception as e:
        logger.error(f"Error in task cleanup: {str(e)}")
        if TRACE_MODE:
            trace_print(f"Cleanup error:\n{traceback.format_exc()}")

@app.post("/update-labs", response_class=JSONResponse)
async def update_labs(request: Request):
    """Start a background task to update lab sections."""
    try:
        # Ensure session exists
        session_id = request.session.get("id")
        if not session_id:
            session_id = secrets.token_urlsafe(16)
            request.session["id"] = session_id
        
        # Get a valid token - first try session-specific tokens
        tokens = await load_tokens_from_redis(session_id)
        
        # If no session tokens, try to find any valid tokens globally
        if not tokens or "access_token" not in tokens or is_token_expired(tokens):
            # Urgent lab caching: use direct token discovery without buffer
            redis_conn = get_redis_sync()
            all_keys = redis_conn.keys("tokens:*")
            
            # For urgent caching, bypass buffer entirely
            current_time = int(time.time())
            
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                try:
                    data = redis_conn.get(key)
                    if data:
                        other_tokens = json.loads(data)
                        if (other_tokens and 
                            "access_token" in other_tokens and 
                            "expires_at" in other_tokens):
                            # URGENT: Direct expiration check - no buffer
                            if other_tokens["expires_at"] > current_time:
                                tokens = other_tokens
                                other_session_id = key_str.replace("tokens:", "")
                                logger.info(f"URGENT LAB CACHING: Using global tokens from session {other_session_id}")
                                break
                except Exception as e:
                    logger.warning(f"Error loading tokens from key {key_str}: {e}")
                    continue
            
            # Final check: ensure token is actually valid (direct expiration only)
            if not tokens or "access_token" not in tokens:
                return JSONResponse({"error": "No valid token found"}, status_code=401)
            
            # Direct expiration check for final validation
            current_time = int(time.time())
            if tokens.get("expires_at", 0) <= current_time:
                return JSONResponse({"error": "No valid token found"}, status_code=401)
        
        token = tokens["access_token"]
        
        # First get student ID
        async with httpx.AsyncClient() as client:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://connect.bracu.ac.bd",
                "Referer": "https://connect.bracu.ac.bd/"
            }
            
            # Get student ID
            portfolios_url = "https://connect.bracu.ac.bd/api/mds/v1/portfolios"
            resp = await client.get(portfolios_url, headers=headers)
            
            if resp.status_code == 401:
                return JSONResponse({"error": "Token expired"}, status_code=401)
            elif resp.status_code != 200:
                return JSONResponse({
                    "error": "Failed to fetch student info. Please try again later."
                }, status_code=resp.status_code)
            
            try:
                data = resp.json()
                if not isinstance(data, list) or not data or "id" not in data[0]:
                    return JSONResponse({"error": "Could not find student id"}, status_code=500)
                student_id = data[0]["id"]
            except Exception as e:
                logger.error(f"Failed to parse student info: {str(e)}")
                return JSONResponse({"error": "Failed to parse student info"}, status_code=500)
            
            # Get student's schedule
            schedule_url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/student/{student_id}/schedules"
            resp = await client.get(schedule_url, headers=headers)
            
            if resp.status_code != 200:
                return JSONResponse({
                    "error": "Failed to fetch schedule. Please try again later."
                }, status_code=resp.status_code)
            
            try:
                sections = resp.json()
            except Exception as e:
                logger.error(f"Failed to parse schedule: {str(e)}")
                return JSONResponse({
                    "error": "Failed to parse schedule data"
                }, status_code=500)
        
        # Generate task ID and start background task
        task_id = secrets.token_urlsafe(8)
        
        # Cancel any existing tasks for this session
        for existing_task_id, task_info in list(BACKGROUND_TASKS.items()):
            if task_info.get("session_id") == session_id and task_info.get("status") == "running":
                task_info["status"] = "cancelled"
                logger.info(f"Cancelled existing task {existing_task_id} for session {session_id}")
        
        # Create new task with session info
        BACKGROUND_TASKS[task_id] = {
            "status": "starting",
            "message": "Initializing...",
            "session_id": session_id,
            "start_time": int(time.time())
        }
        
        # Start the task
        background_task = asyncio.create_task(
            update_all_labs_background(token, sections, task_id),
            name=f"update_labs_{task_id}"
        )
        
        # Add to active tasks set
        ACTIVE_TASKS.add(background_task)
        
        # Add task cleanup callback
        def task_done_callback(task):
            try:
                # Remove from active tasks
                ACTIVE_TASKS.discard(task)
                
                # Handle task completion
                if task.cancelled():
                    logger.warning(f"Task {task_id} was cancelled")
                    BACKGROUND_TASKS[task_id] = {
                        "status": "cancelled",
                        "message": "Task was cancelled",
                        "completion_time": int(time.time())
                    }
                else:
                    exc = task.exception()
                    if exc:
                        logger.error(f"Task {task_id} failed with exception: {exc}")
                        BACKGROUND_TASKS[task_id] = {
                            "status": "error",
                            "message": f"Task failed: {str(exc)}",
                            "completion_time": int(time.time())
                        }
                    elif task_id in BACKGROUND_TASKS and BACKGROUND_TASKS[task_id].get("status") not in ["completed", "error", "cancelled"]:
                        BACKGROUND_TASKS[task_id] = {
                            "status": "completed",
                            "message": "Task completed",
                            "completion_time": int(time.time())
                        }
            except Exception as e:
                logger.error(f"Error in task completion callback: {str(e)}")
            finally:
                # Always clean up task resources
                asyncio.create_task(cleanup_task(task_id))
        
        background_task.add_done_callback(task_done_callback)
        
        return JSONResponse({"task_id": task_id})
        
    except Exception as e:
        logger.error(f"Error starting update task: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        return JSONResponse({
            "error": "An unexpected error occurred. Please try again later.",
            "details": str(e) if DEBUG_MODE else None
        }, status_code=500)

async def update_all_labs_background(token: str, sections: list, task_id: str):
    """Background task to update all lab sections."""
    client = None
    processed = 0
    total = 0
    new_labs = 0
    failed = 0
    
    try:
        # Add task to active tasks
        current_task = asyncio.current_task()
        if current_task:
            ACTIVE_TASKS.add(current_task)
        
        async with get_task_lock(task_id):
            global lab_cache
            if not lab_cache:
                lab_cache = await load_lab_cache_from_redis() or {}
            
            # Filter sections that need to be checked for labs
            sections_to_check = [
                section for section in sections 
                if section.get("sectionId") 
                and str(section.get("sectionId")) not in lab_cache
                and section.get("sectionType") != "LAB"
                and (section.get("sectionType") == "THEORY" or section.get("sectionType") == "OTHER")
            ]
            
            if not sections_to_check:
                debug_print("No new sections to check for labs")
                BACKGROUND_TASKS[task_id] = {
                    "status": "completed", 
                    "message": "No new sections to check",
                    "completion_time": int(time.time())
                }
                return
            
            total = len(sections_to_check)
            debug_print(f"Starting lab update for {total} sections")
            
            # Create client with explicit timeout and limits
            client = httpx.AsyncClient(
                timeout=30.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
            
            try:
                for section in sections_to_check:
                    # Check if task was cancelled
                    if task_id not in BACKGROUND_TASKS or BACKGROUND_TASKS[task_id].get("status") == "cancelled":
                        logger.warning(f"Task {task_id} was cancelled during processing")
                        return
                    
                    # Check if current task is being cancelled
                    if current_task and current_task.cancelled():
                        logger.warning(f"Task {task_id} received cancellation signal")
                        raise asyncio.CancelledError()
                        
                    section_id = str(section.get("sectionId"))
                    success = False
                    
                    # Try up to 3 times for each section
                    for attempt in range(3):
                        try:
                            # Get a fresh token for each attempt
                            current_token = await get_latest_valid_token()
                            if not current_token:
                                logger.error("No valid token available")
                                BACKGROUND_TASKS[task_id] = {
                                    "status": "error",
                                    "message": f"Token expired. Processed {processed}/{total} sections. Found {new_labs} labs. Failed: {failed}.",
                                    "completion_time": int(time.time())
                                }
                                return
                            
                            headers = {
                                "Accept": "application/json",
                                "Authorization": f"Bearer {current_token}",
                                "User-Agent": "Mozilla/5.0",
                                "Origin": "https://connect.bracu.ac.bd",
                                "Referer": "https://connect.bracu.ac.bd/"
                            }
                            
                            url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/{section_id}/details"
                            debug_print(f"Fetching details for section {section_id}, attempt {attempt + 1}")
                            
                            # Use shield to prevent cancellation during request
                            resp = await asyncio.shield(client.get(url, headers=headers, timeout=10.0))
                                
                            if resp.status_code == 200:
                                details_data = resp.json()
                                child_section = details_data.get("childSection")
                                if child_section:
                                    # Parse sectionSchedule if it's a string
                                    lab_schedule = child_section.get("sectionSchedule")
                                    if isinstance(lab_schedule, str):
                                        try:
                                            lab_schedule = json.loads(lab_schedule)
                                        except:
                                            lab_schedule = None
                                    
                                    lab_info = {
                                        "sectionId": section_id,
                                        "labSectionId": child_section.get("sectionId"),
                                        "labCourseCode": child_section.get("courseCode"),
                                        "labFaculties": child_section.get("faculties"),
                                        "labName": child_section.get("sectionName"),
                                        "labRoomName": child_section.get("roomName"),
                                        "labSchedules": lab_schedule
                                    }
                                    lab_cache[str(section_id)] = lab_info
                                    new_labs += 1
                                    debug_print(f"Found lab section for {section_id}: {child_section.get('courseCode')} {child_section.get('sectionName')}")
                                success = True
                                break
                            elif resp.status_code == 401:
                                # Token expired, will retry with new token
                                logger.warning(f"Token expired during section {section_id} check, attempt {attempt + 1}")
                                await asyncio.sleep(1)  # Small delay before retry
                                continue
                            else:
                                logger.error(f"Failed to get section {section_id} details: HTTP {resp.status_code}")
                                if DEBUG_MODE:
                                    debug_print(f"Response content: {resp.text}")
                                await asyncio.sleep(1)
                        
                        except (httpx.RequestError, asyncio.TimeoutError) as e:
                            logger.error(f"Network error processing section {section_id}: {str(e)}")
                            if TRACE_MODE:
                                trace_print(f"Network error details:\n{traceback.format_exc()}")
                            await asyncio.sleep(1)
                            continue
                        except Exception as e:
                            logger.error(f"Error processing section {section_id}: {str(e)}")
                            if DEBUG_MODE or TRACE_MODE:
                                trace_print(f"Stack trace:\n{traceback.format_exc()}")
                            await asyncio.sleep(1)
                            continue
                    
                    if not success:
                        failed += 1
                        debug_print(f"Failed to process section {section_id} after all attempts")
                    
                    processed += 1
                    BACKGROUND_TASKS[task_id] = {
                        "status": "running",
                        "progress": (processed / total) * 100,
                        "message": f"Processed {processed}/{total} sections. Found {new_labs} labs. Failed: {failed}."
                    }
                    
                    # Small delay between sections
                    await asyncio.sleep(0.2)
                
                # Save final results to Redis and deduplicate
                if new_labs > 0:
                    final_count = await deduplicate_lab_cache()
                    logger.info(f"Lab cache deduplicated to {final_count} unique lab sections")
                    await save_lab_cache_to_redis(lab_cache)
                    debug_print(f"Added {new_labs} new lab sections to Redis cache")
                
                BACKGROUND_TASKS[task_id] = {
                    "status": "completed",
                    "message": f"Completed. Added {new_labs} new lab sections to cache. Failed to process {failed} sections.",
                    "completion_time": int(time.time())
                }
                
            except asyncio.CancelledError:
                logger.warning(f"Task {task_id} was cancelled during section processing")
                raise
            finally:
                # Ensure client is closed
                if client:
                    await client.aclose()
                
                # Remove task from active tasks
                if current_task:
                    ACTIVE_TASKS.discard(current_task)
    
    except asyncio.CancelledError:
        logger.warning(f"Task {task_id} was cancelled")
        BACKGROUND_TASKS[task_id] = {
            "status": "cancelled",
            "message": f"Task cancelled. Processed {processed}/{total} sections. Found {new_labs} labs. Failed: {failed}.",
            "completion_time": int(time.time())
        }
        raise
    except Exception as e:
        logger.error(f"Error in background task: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        BACKGROUND_TASKS[task_id] = {
            "status": "error", 
            "message": str(e),
            "completion_time": int(time.time())
        }
        raise
    finally:
        # Always clean up task resources
        await cleanup_task(task_id)

# Add shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up any remaining tasks on shutdown."""
    debug_print("Server shutting down, cleaning up tasks...")
    
    # Cancel all active tasks
    for task in ACTIVE_TASKS:
        if not task.done():
            task.cancel()
    
    # Wait for all tasks to complete
    if ACTIVE_TASKS:
        await asyncio.gather(*ACTIVE_TASKS, return_exceptions=True)
    
    # Clear task collections
    ACTIVE_TASKS.clear()
    BACKGROUND_TASKS.clear()
    TASK_LOCKS.clear()

async def validate_token_live(access_token: str) -> dict:
    """Validate token by calling the BRACU Connect API portfolios endpoint."""
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "Mozilla/5.0",
            "Origin": "https://connect.bracu.ac.bd",
            "Referer": "https://connect.bracu.ac.bd/"
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://connect.bracu.ac.bd/api/mds/v1/portfolios",
                headers=headers,
                timeout=10.0
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data and len(data) > 0 and "id" in data[0]:
                    return {
                        "valid": True,
                        "student_id": data[0].get("studentId"),
                        "full_name": data[0].get("fullName"),
                        "portfolio_id": data[0].get("id"),
                        "timestamp": int(time.time())
                    }
                else:
                    return {"valid": False, "error": "Invalid response format"}
            elif resp.status_code == 401:
                return {"valid": False, "error": "Token expired or invalid"}
            else:
                return {"valid": False, "error": f"HTTP {resp.status_code}"}
                
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        return {"valid": False, "error": str(e)}

@app.get("/token-status")
async def get_token_status(request: Request):
    """Get current token status with live validation."""
    session_id = request.session.get("id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        request.session["id"] = session_id
        return {"valid": False, "error": "No session"}
    
    tokens = await load_tokens_from_redis(session_id)
    if not tokens or not tokens.get("access_token"):
        return {"valid": False, "error": "No tokens found"}
    
    # Validate token live
    validation_result = await validate_token_live(tokens["access_token"])
    
    # Update cache with validation timestamp
    if validation_result["valid"]:
        tokens["last_validation"] = validation_result["timestamp"]
        await save_tokens_to_redis(session_id, tokens)
    
    return validation_result

@app.get("/token-status/continuous")
async def continuous_token_check(request: Request):
    """Continuous token validation endpoint for real-time updates."""
    session_id = request.session.get("id")
    if not session_id:
        session_id = secrets.token_urlsafe(16)
        request.session["id"] = session_id
        return {"valid": False, "error": "No session"}
    
    tokens = await load_tokens_from_redis(session_id)
    if not tokens or not tokens.get("access_token"):
        return {"valid": False, "error": "No tokens found"}
    
    validation_result = await validate_token_live(tokens["access_token"])
    return validation_result

@app.get("/global-token-status")
async def get_global_token_status():
    """Get global token status - shows if any valid tokens exist in Redis without exposing actual tokens."""
    try:
        redis_conn = get_redis_sync()
        
        # Get all token keys
        token_keys = redis_conn.keys("tokens:*")
        if not token_keys:
            return {
                "has_active_tokens": False,
                "message": "No tokens found in Redis",
                "total_sessions": 0,
                "active_sessions": 0,
                "last_updated": int(time.time())
            }
        
        total_sessions = len(token_keys)
        active_sessions = 0
        earliest_expiry = None
        latest_expiry = None
        
        # Check each token for validity
        for key in token_keys:
            try:
                tokens_str = redis_conn.get(key)
                if tokens_str:
                    tokens = json.loads(tokens_str)
                    if tokens and "access_token" in tokens:
                        # Check if token is valid and not expired
                        if not is_token_expired(tokens):
                            active_sessions += 1
                            
                            # Track expiry times
                            if "expires_at" in tokens:
                                expiry = tokens["expires_at"]
                                if earliest_expiry is None or expiry < earliest_expiry:
                                    earliest_expiry = expiry
                                if latest_expiry is None or expiry > latest_expiry:
                                    latest_expiry = expiry
            except (json.JSONDecodeError, Exception):
                continue
        
        has_active_tokens = active_sessions > 0
        
        # Calculate uptime and remaining time
        uptime_seconds = None
        uptime_display = None
        remaining_display = None
        
        # Find the earliest created active token for uptime calculation
        earliest_created = None
        if has_active_tokens:
            for key in token_keys:
                try:
                    tokens_str = redis_conn.get(key)
                    if tokens_str:
                        tokens = json.loads(tokens_str)
                        if tokens and "access_token" in tokens and not is_token_expired(tokens):
                            created_at = tokens.get("activated_at", tokens.get("created_at"))
                            if created_at is None:
                                # Fallback to current time if no creation timestamp
                                created_at = int(time.time())
                            if earliest_created is None or created_at < earliest_created:
                                earliest_created = created_at
                except:
                    continue
        
        if earliest_created:
            now = int(time.time())
            uptime_seconds = max(0, now - earliest_created)
            
            # Format uptime display
            uptime_parts = []
            if uptime_seconds >= 86400:
                days = uptime_seconds // 86400
                uptime_parts.append(f"{days}d")
                hours = (uptime_seconds % 86400) // 3600
                if hours > 0:
                    uptime_parts.append(f"{hours}h")
            elif uptime_seconds >= 3600:
                hours = uptime_seconds // 3600
                uptime_parts.append(f"{hours}h")
                minutes = (uptime_seconds % 3600) // 60
                if minutes > 0:
                    uptime_parts.append(f"{minutes}m")
            else:
                minutes = uptime_seconds // 60
                if minutes > 0:
                    uptime_parts.append(f"{minutes}m")
                seconds = uptime_seconds % 60
                uptime_parts.append(f"{seconds}s")
            
            uptime_display = " ".join(uptime_parts)
        
        if earliest_expiry:
            now = int(time.time())
            remaining = max(0, earliest_expiry - now)
            if remaining > 0:
                days, remainder = divmod(remaining, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                parts = []
                if days:
                    parts.append(f"{days}d")
                if hours:
                    parts.append(f"{hours}h")
                if minutes:
                    parts.append(f"{minutes}m")
                if seconds or not parts:
                    parts.append(f"{seconds}s")
                
                remaining_display = " ".join(parts)
        
        return {
            "has_active_tokens": has_active_tokens,
            "message": f"{active_sessions} active tokens found across {total_sessions} sessions" if has_active_tokens else "No active tokens found",
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "earliest_token_expires": earliest_expiry,
            "latest_token_expires": latest_expiry,
            "uptime_seconds": uptime_seconds,
            "uptime_display": uptime_display,
            "remaining_time_display": remaining_display,
            "last_updated": int(time.time())
        }
        
    except Exception as e:
        logger.error(f"Error checking global token status: {str(e)}")
        return {
            "has_active_tokens": False,
            "message": "Error checking token status",
            "error": str(e),
            "last_updated": int(time.time())
        }

@app.get("/update-labs/status/{task_id}", response_class=JSONResponse)
async def get_update_status(task_id: str):
    """Get the status of a background lab update task."""
    # Clean up old tasks first
    await cleanup_old_tasks()
    
    if task_id not in BACKGROUND_TASKS:
        return JSONResponse({
            "status": "not_found",
            "message": "Task not found or expired. Please start a new update."
        }, status_code=404)
    
    task_info = BACKGROUND_TASKS[task_id]
    
    # Add completion time when task finishes
    if (task_info.get("status") in ["completed", "error", "cancelled"] and 
        "completion_time" not in task_info):
        task_info["completion_time"] = int(time.time())
    
    # Calculate time remaining before cleanup if task is done
    if task_info.get("completion_time"):
        remaining = max(0, (task_info["completion_time"] + TASK_RETENTION_TIME) - int(time.time()))
        task_info["cleanup_in"] = remaining
    
    return JSONResponse(task_info)

# Vercel handler for serverless deployment

# =============================================================
# ConnectAPI - BRACU Connect API Server
# Developed by Wasif Faisal to support https://routinez.vercel.app/
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
from redis import asyncio as aioredis
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
SHOW_VIEW_TOKENS = False  # Set to False to hide the View Tokens button

# Get OAuth2 credentials from environment variables
OAUTH_CLIENT_ID = os.getenv('OAUTH_CLIENT_ID', 'connect-portal')
OAUTH_CLIENT_SECRET = os.getenv('OAUTH_CLIENT_SECRET', '')
OAUTH_TOKEN_URL = os.getenv('OAUTH_TOKEN_URL', 'https://sso.bracu.ac.bd/realms/bracu/protocol/openid-connect/token')

# Development mode - set to True for local development
DEV_MODE = True

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

# Create FastAPI app first
app = FastAPI()

# Add middleware in correct order
app.add_middleware(SessionMiddleware, secret_key="super-secret-session-key")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL] if not DEV_MODE else ["*"],
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

@app.middleware("http")
async def session_error_handler(request: Request, call_next):
    response = await call_next(request)
    return response

# Upstash Redis config
REDIS_URL = os.environ.get("REDIS_URL") or "rediss://default:@willing-husky-43244.upstash.io:6379"

# Redis client instance cache to avoid creating new connections repeatedly
_redis_client_cache = None

def get_redis_sync():
    # Use Upstash Redis with proper configuration
    global _redis_client_cache
    
    # Return cached client if available
    if _redis_client_cache is not None:
        return _redis_client_cache
        
    try:
        from upstash_redis import Redis
        # Extract URL and token for Upstash Redis
        _redis_client_cache = Redis(
            url="https://willing-husky-43244.upstash.io",
            token="AajsAAIncDFjY2Q4YWQzMjYwODY0ZWJkYTZjYzYyNGNhNjUwODkzZHAxNDMyNDQ"
        )
        return _redis_client_cache
    except ImportError:
        # Fallback to aioredis if upstash_redis is not available
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        _redis_client_cache = aioredis.from_url(REDIS_URL, decode_responses=True, ssl_context=ssl_context)
        return _redis_client_cache

async def get_redis():
    # This is an async wrapper for get_redis_sync
    # It returns the same Redis client but is properly defined as async
    # to avoid "object Redis can't be used in 'await' expression" errors
    return get_redis_sync()

# Cache for method type checks to avoid repeated attribute lookups
_redis_method_cache = {}

async def redis_get(redis_conn, key):
    """Helper function to handle both sync and async Redis get operations."""
    conn_id = id(redis_conn)
    if conn_id not in _redis_method_cache:
        _redis_method_cache[conn_id] = {}
    
    if 'get' not in _redis_method_cache[conn_id]:
        _redis_method_cache[conn_id]['get'] = hasattr(redis_conn.get, "__await__")
    
    if _redis_method_cache[conn_id]['get']:
        return await redis_conn.get(key)
    else:
        return redis_conn.get(key)

async def redis_set(redis_conn, key, value):
    """Helper function to handle both sync and async Redis set operations."""
    conn_id = id(redis_conn)
    if conn_id not in _redis_method_cache:
        _redis_method_cache[conn_id] = {}
    
    if 'set' not in _redis_method_cache[conn_id]:
        _redis_method_cache[conn_id]['set'] = hasattr(redis_conn.set, "__await__")
    
    if _redis_method_cache[conn_id]['set']:
        return await redis_conn.set(key, value)
    else:
        return redis_conn.set(key, value)

async def redis_keys(redis_conn, pattern):
    """Helper function to handle both sync and async Redis keys operations."""
    conn_id = id(redis_conn)
    if conn_id not in _redis_method_cache:
        _redis_method_cache[conn_id] = {}
    
    if 'keys' not in _redis_method_cache[conn_id]:
        _redis_method_cache[conn_id]['keys'] = hasattr(redis_conn.keys, "__await__")
    
    if _redis_method_cache[conn_id]['keys']:
        return await redis_conn.keys(pattern)
    else:
        return redis_conn.keys(pattern)

async def redis_delete(redis_conn, key):
    """Helper function to handle both sync and async Redis delete operations."""
    conn_id = id(redis_conn)
    if conn_id not in _redis_method_cache:
        _redis_method_cache[conn_id] = {}
    
    if 'delete' not in _redis_method_cache[conn_id]:
        _redis_method_cache[conn_id]['delete'] = hasattr(redis_conn.delete, "__await__")
    
    if _redis_method_cache[conn_id]['delete']:
        return await redis_conn.delete(key)
    else:
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
            else:
                tokens["expires_at"] = now + 300
                tokens["expires_in"] = 300
                logger.warning("No expiration found in JWT, using default 5 minutes")
        
        # Always set refresh token expiration to 30 minutes from now if we have a refresh token
        if "refresh_token" in tokens:
            tokens["refresh_expires_at"] = now + (30 * 60)  # 30 minutes
        
        # Save tokens with expiration
        key = f"tokens:{session_id}"
        redis_conn.set(key, json.dumps(tokens))
        
        # Set key expiration to match the refresh token expiration
        redis_conn.expire(key, 30 * 60)  # 30 minutes
        
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
        return _save_tokens_to_redis_sync(session_id, tokens)
    except Exception as e:
        logger.error(f"Error in async save_tokens_to_redis: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
        raise

@with_redis_retry
def _load_tokens_from_redis_sync(session_id):
    """Load tokens from Redis with validation."""
    try:
        redis_conn = get_redis_sync()
        key = f"tokens:{session_id}"
        data = redis_conn.get(key)
        
        if data:
            tokens = json.loads(data)
            # Validate token expiration
            if not is_token_expired(tokens):
                debug_print(f"Valid tokens loaded from Redis for session {session_id}")
                return tokens
            else:
                debug_print(f"Expired tokens found for session {session_id}, attempting refresh")
                if "refresh_token" in tokens:
                    try:
                        new_tokens = asyncio.run(refresh_access_token(tokens["refresh_token"]))
                        if new_tokens:
                            _save_tokens_to_redis_sync(session_id, new_tokens)
                            return new_tokens
                    except Exception as e:
                        logger.error(f"Token refresh failed: {str(e)}")
                        if DEV_MODE:
                            logger.error(traceback.format_exc())
                
                # If we get here, tokens are expired and refresh failed
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
    """Async wrapper for load_tokens_from_redis function."""
    try:
        return _load_tokens_from_redis_sync(session_id)
    except Exception as e:
        logger.error(f"Error in async load_tokens_from_redis: {str(e)}")
        if DEBUG_MODE or TRACE_MODE:
            trace_print(f"Stack trace:\n{traceback.format_exc()}")
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
        return _save_student_schedule_sync(student_id, schedule)
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
        return _load_student_schedule_sync(student_id)
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

def is_token_expired(tokens, buffer=60):
    """Check if tokens are expired with a buffer time."""
    if not tokens:
        return True
    now = int(time.time())
    # Check access token expiration
    if "expires_at" in tokens:
        if now + buffer >= tokens["expires_at"]:
            return True
    return False

@with_redis_retry
async def get_latest_valid_token():
    """Get the most recent valid token from Redis, attempting to refresh if needed."""
    try:
        redis_conn = await get_redis()
            
        # Get all token keys
        token_keys = await redis_keys(redis_conn, "tokens:*")
        if not token_keys:
            logger.warning("No tokens found in Redis")
            return None
            
        latest_token = None
        latest_expiry = 0
        needs_refresh = True
        session_id = None
        
        # First try to find the most recent valid token
        for key in token_keys:
            tokens_str = await redis_get(redis_conn, key)
            if tokens_str:
                try:
                    tokens = json.loads(tokens_str)
                    if "expires_at" in tokens:
                        # If this token expires later than our current latest, update it
                        if tokens["expires_at"] > latest_expiry:
                            latest_token = tokens
                            latest_expiry = tokens["expires_at"]
                            needs_refresh = is_token_expired(tokens)
                            session_id = key.split(":")[-1]
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
                new_tokens = await refresh_access_token(latest_token["refresh_token"])
                if new_tokens and "access_token" in new_tokens:
                    # Save the refreshed tokens
                    await save_tokens_to_redis(session_id, new_tokens)
                    logger.info("Successfully refreshed token")
                    return new_tokens.get("access_token")
            except Exception as e:
                logger.error(f"Error refreshing token: {str(e)}")
                # Delete the expired/invalid tokens
                await redis_delete(redis_conn, f"tokens:{session_id}")
        
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
    try:
        tokens = await load_tokens_from_redis(session_id)
        if tokens and "access_token" in tokens and not is_token_expired(tokens):
            token = tokens["access_token"]
            
            # Check section count if we have a valid token
            async with httpx.AsyncClient() as client:
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0",
                    "Origin": "https://connect.bracu.ac.bd",
                    "Referer": "https://connect.bracu.ac.bd/"
                }
                
                url = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections"
                resp = await client.get(url, headers=headers)
                
                if resp.status_code == 200:
                    sections = resp.json()
                    has_changed, current, stored = await check_section_changes(sections)
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

    // Check for existing task on page load
    window.onload = function() {{
        const urlParams = new URLSearchParams(window.location.search);
        const taskId = urlParams.get('task_id');
        if (taskId) {{
            currentTaskId = taskId;
            document.getElementById('updateStatus').style.display = 'block';
            startStatusCheck();
        }}
    }};
    </script></head><body>
    <div class='container'>
        <h1>BRACU Schedule Viewer</h1>
        <div class='desc'>A simple client to view your BRACU Connect schedule.<br>Session-based, no password required.</div>
        <div class='button-container'>
            <a class='button' href='/enter-tokens'>Enter Tokens</a>
            {f"<a class='button' href='/mytokens'>View Tokens</a>" if SHOW_VIEW_TOKENS else ""}
            <a class='button' href='/raw-schedule'>View Raw Schedule</a>
            <button class='button update' onclick='updateLabs()'>Update Lab Cache</button>
        </div>
        {section_status_display}
        <div id='updateStatus'></div>
        <div class='session-id'>Session: {session_id}</div>
        <div class='network-uptime'>{network_uptime_display}</div>
        <div class='token-remaining'>{token_remaining_display}</div>
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
async def view_tokens(request: Request, session_id: str = None):
    """View tokens for the current session."""
    try:
        current_session = request.session.get("id")
        if not current_session:
            # No session, redirect to home
            return RedirectResponse("/", status_code=302)
        
        # If a specific session_id is requested, verify it matches the current session
        if session_id and session_id != current_session:
            return HTMLResponse("""
                <html><head><title>Error</title>
                <style>
                body { font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }
                .container { max-width: 520px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; text-align: center; }
                .error { color: #e53e3e; margin-bottom: 18px; }
                .back { display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }
                .back:hover { text-decoration: underline; }
                </style></head><body>
                <div class='container'>
                    <div class='error'>You can only view tokens for your own session.</div>
                    <a class='back' href='/'>Back to Home</a>
                </div></body></html>
            """, status_code=403)

        # Load tokens for the current session
        redis_conn = await get_redis()
        tokens = await load_tokens_from_redis(current_session)
        
        # If no tokens found in current session, try to get the latest valid token
        if not tokens:
            latest_token = await get_latest_valid_token()
            if latest_token:
                # Create new tokens object with the latest token
                tokens = {
                    "access_token": latest_token,
                    "expires_at": int(time.time()) + 300,  # 5 minutes default
                    "refresh_expires_at": int(time.time()) + 1800  # 30 minutes default
                }
                # Save these tokens to the current session
                await save_tokens_to_redis(current_session, tokens)
                logger.info(f"Saved latest valid token to session {current_session}")
        
        # Calculate token expiration times if tokens exist
        token_info = ""
        if tokens:
            now = int(time.time())
            access_expires_in = max(0, tokens.get("expires_at", 0) - now)
            refresh_expires_in = max(0, tokens.get("refresh_expires_at", 0) - now)
            
            # If tokens are expired, try to refresh them
            if access_expires_in <= 0 and "refresh_token" in tokens:
                try:
                    new_tokens = await refresh_access_token(tokens["refresh_token"])
                    if new_tokens:
                        await save_tokens_to_redis(current_session, new_tokens)
                        tokens = new_tokens
                        access_expires_in = max(0, tokens.get("expires_at", 0) - now)
                        refresh_expires_in = max(0, tokens.get("refresh_expires_at", 0) - now)
                        logger.info(f"Refreshed tokens for session {current_session}")
                except Exception as e:
                    logger.error(f"Token refresh failed: {str(e)}")
            
            token_info = f"""
            <div class='token-info'>
                <div class='expiry'>Access token expires in: {access_expires_in} seconds</div>
                <div class='expiry'>Refresh token expires in: {refresh_expires_in} seconds</div>
            </div>
            """

        html_content = f"""
        <html><head><title>My Tokens</title>
        <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f5f5f5; margin: 0; }}
        .container {{ max-width: 520px; margin: 60px auto; background: #fff; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 36px 28px; }}
        h2 {{ color: #2d3748; margin-bottom: 18px; }}
        pre {{ background: #f7fafc; border-radius: 6px; padding: 18px; font-size: 1em; color: #2d3748; overflow-x: auto; }}
        .msg {{ color: #e53e3e; margin-bottom: 18px; }}
        .back {{ display: block; margin-top: 18px; color: #3182ce; text-decoration: none; }}
        .back:hover {{ text-decoration: underline; }}
        .token-info {{ margin: 12px 0; padding: 12px; background: #ebf8ff; border-radius: 6px; }}
        .expiry {{ color: #2b6cb0; margin: 4px 0; }}
        .session {{ font-size: 0.9em; color: #718096; margin-top: 12px; }}
        </style></head><body>
        <div class='container'>
            <h2>Your Tokens</h2>
            {token_info if tokens else '<div class="msg">No tokens found for your session.</div>'}
            {('<pre>' + json.dumps(tokens, indent=2) + '</pre>') if tokens else ''}
            <div class='session'>Session ID: {current_session}</div>
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
        redis_conn = await get_redis()
        count = await redis_conn.get("total_section_count")
        return int(count) if count else None
    except Exception as e:
        logger.error(f"Error getting section count from Redis: {str(e)}")
        return None

async def save_section_count_to_redis(count: int):
    """Save the current section count to Redis."""
    try:
        redis_conn = await get_redis()
        await redis_conn.set("total_section_count", str(count))
        logger.info(f"Saved section count to Redis: {count}")
    except Exception as e:
        logger.error(f"Error saving section count to Redis: {str(e)}")

# Initialize empty lab cache
lab_cache = {}

async def save_lab_cache_to_redis(lab_data: dict):
    """Save lab section data to Redis."""
    try:
        redis_conn = await get_redis()
        await redis_set(redis_conn, "lab_cache", json.dumps(lab_data))
        logger.info(f"Saved {len(lab_data)} lab sections to Redis cache")
        return True
    except Exception as e:
        logger.error(f"Error saving lab cache to Redis: {str(e)}")
        return False

async def load_lab_cache_from_redis():
    """Load lab section data from Redis."""
    try:
        redis_conn = await get_redis()
        data = await redis_get(redis_conn, "lab_cache")
        if data:
            return json.loads(data)
        return {}
    except Exception as e:
        logger.error(f"Error loading lab cache from Redis: {str(e)}")
        return {}

async def initialize_lab_cache():
    """Initialize lab cache from Redis."""
    global lab_cache
    try:
        lab_cache = await load_lab_cache_from_redis()
        logger.info(f"Loaded {len(lab_cache)} lab sections from Redis cache")
    except Exception as e:
        logger.warning(f"Could not load lab cache from Redis: {str(e)}. Starting with empty lab cache.")

# Initialize lab cache on startup
@app.on_event("startup")
async def startup_event():
    await initialize_lab_cache()

async def update_lab_cache(token: str, sections: list):
    """Update lab cache with new lab sections found in the provided sections list."""
    global lab_cache
    updated_lab_data = []
    
    # Ensure lab_cache is loaded from Redis if it's empty
    if not lab_cache:
        lab_cache = await load_lab_cache_from_redis() or {}
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
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://connect.bracu.ac.bd",
                "Referer": "https://connect.bracu.ac.bd/"
            }
            
            # Process sections in batches
            for i in range(0, len(sections_to_check), batch_size):
                batch = sections_to_check[i:i+batch_size]
                tasks = []
                
                for section in batch:
                    section_id = str(section.get("sectionId"))
                    # Skip if already in cache
                    if section_id in lab_cache:
                        continue
                        
                    url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/{section_id}/details"
                    tasks.append(asyncio.create_task(client.get(url, headers=headers)))
                
                # Wait for all requests in this batch to complete
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process the responses
                for j, response in enumerate(responses):
                    if j >= len(batch):  # Safety check
                        break
                        
                    section = batch[j]
                    section_id = str(section.get("sectionId"))
                    
                    try:
                        if isinstance(response, Exception):
                            logger.error(f"Error fetching section details for section {section_id}: {str(response)}")
                            continue
                            
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
                                
                                lab_info = {
                                    "sectionId": section_id,
                                    "labSectionId": child_section.get("sectionId"),
                                    "labCourseCode": child_section.get("courseCode"),
                                    "labFaculties": child_section.get("faculties"),
                                    "labName": child_section.get("sectionName"),
                                    "labRoomName": child_section.get("roomName"),
                                    "labSchedules": lab_schedule
                                }
                                updated_lab_data.append(lab_info)
                                
                                # Update the section in the original list using the lookup dictionary
                                if section_id in section_dict:
                                    sections[section_dict[section_id]].update(lab_info)
                                
                                # Update the lab cache
                                lab_cache[section_id] = lab_info
                    except Exception as e:
                        logger.error(f"Error processing section details for section {section_id}: {str(e)}")
                        continue
    
        # Update Redis with any new lab sections found
        if updated_lab_data:
            try:
                # Save updated lab cache to Redis
                await save_lab_cache_to_redis(lab_cache)
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

# Cache for raw schedule to avoid frequent API calls
_schedule_cache = {}
_schedule_cache_expiry = {}
SCHEDULE_CACHE_TTL = 60  # Cache for 60 seconds

@app.get("/raw-schedule", response_class=JSONResponse)
async def raw_schedule(request: Request):
    """Get the raw schedule data. Public endpoint using most recent valid token or latest cached schedule."""
    try:
        # Check if we have a valid cached response
        cache_key = "latest_schedule"
        current_time = time.time()
        
        if cache_key in _schedule_cache and _schedule_cache_expiry.get(cache_key, 0) > current_time:
            logger.info("Returning cached schedule response")
            return JSONResponse(_schedule_cache[cache_key])
            
        async with get_schedule_lock():  # Use the new lock manager
            session_id = request.session.get("id")
            # Try to get a valid token
            if session_id:
                tokens = await load_tokens_from_redis(session_id)
                if tokens and "access_token" in tokens and not is_token_expired(tokens):
                    token = tokens["access_token"]
                else:
                    token = await get_latest_valid_token()
            else:
                token = await get_latest_valid_token()

            redis_conn = await get_redis()

            if not token:
                keys = await redis_keys(redis_conn, "student_schedule:*")
                if keys:
                    latest_key = sorted(keys)[-1]
                    cached_schedule = await redis_get(redis_conn, latest_key)
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
                            "data": schedule_data
                        }
                        
                        # Cache the response
                        _schedule_cache[cache_key] = response_data
                        _schedule_cache_expiry[cache_key] = current_time + SCHEDULE_CACHE_TTL
                        
                        return JSONResponse(response_data)
                return JSONResponse({"error": "No valid token or cached schedule available"}, status_code=503)

            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {token}",
                "User-Agent": "Mozilla/5.0",
                "Origin": "https://connect.bracu.ac.bd",
                "Referer": "https://connect.bracu.ac.bd/"
            }
            
            # Get student ID and schedule in one go
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Get student ID and schedule concurrently
                portfolios_url = "https://connect.bracu.ac.bd/api/mds/v1/portfolios"
                
                # Start the request for student ID
                student_id_task = asyncio.create_task(client.get(portfolios_url, headers=headers))
                
                # Wait for student ID request to complete
                resp = await student_id_task
                
                if resp.status_code == 401:
                    logger.warning("Token unauthorized")
                    if session_id:
                        await redis_delete(redis_conn, f"tokens:{session_id}")
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
                cached_schedule = await load_student_schedule(student_id)
                if cached_schedule:
                    # Use cached schedule as a fallback if API call fails
                    fallback_schedule = cached_schedule
                else:
                    fallback_schedule = None

                # Get schedule and seat status concurrently
                schedule_url = f"https://connect.bracu.ac.bd/api/adv/v1/advising/sections/student/{student_id}/schedules"
                seat_status_url = "https://connect.bracu.ac.bd/api/adv/v1/advising/sections/seat-status"
                
                schedule_task = asyncio.create_task(client.get(schedule_url, headers=headers))
                seat_status_task = asyncio.create_task(client.get(seat_status_url, headers=headers))
                
                # Wait for both tasks to complete
                schedule_resp, seat_status_resp = await asyncio.gather(schedule_task, seat_status_task, return_exceptions=True)
                
                # Handle schedule response
                if isinstance(schedule_resp, Exception):
                    logger.error(f"Error fetching schedule: {str(schedule_resp)}")
                    if fallback_schedule:
                        response_data = {
                            "cached": True,
                            "data": fallback_schedule
                        }
                        # Cache the response
                        _schedule_cache[cache_key] = response_data
                        _schedule_cache_expiry[cache_key] = current_time + SCHEDULE_CACHE_TTL
                        return JSONResponse(response_data)
                    return JSONResponse({"error": "Failed to fetch schedule"}, status_code=500)
                
                if schedule_resp.status_code != 200:
                    if schedule_resp.status_code == 401:
                        logger.warning("Token unauthorized")
                        if session_id:
                            await redis_delete(redis_conn, f"tokens:{session_id}")
                        return JSONResponse({"error": "Token expired, please refresh page"}, status_code=401)
                    
                    if fallback_schedule:
                        response_data = {
                            "cached": True,
                            "data": fallback_schedule
                        }
                        # Cache the response
                        _schedule_cache[cache_key] = response_data
                        _schedule_cache_expiry[cache_key] = current_time + SCHEDULE_CACHE_TTL
                        return JSONResponse(response_data)
                    
                    return JSONResponse({
                        "error": "Failed to fetch schedule",
                        "status_code": schedule_resp.status_code
                    }, status_code=schedule_resp.status_code)
                
                schedule_data = schedule_resp.json()
                
                # Process seat status response
                booked_seats = {}
                if not isinstance(seat_status_resp, Exception) and seat_status_resp.status_code == 200:
                    try:
                        booked_seats = seat_status_resp.json()
                    except Exception as e:
                        logger.error(f"Error parsing seat status: {str(e)}")
                
                # Ensure lab_cache is loaded
                global lab_cache
                if not lab_cache:
                    lab_cache = await load_lab_cache_from_redis() or {}
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
                await save_student_schedule(student_id, schedule_data)
                
                response_data = {
                    "cached": False,
                    "data": schedule_data
                }
                
                # Cache the response
                _schedule_cache[cache_key] = response_data
                _schedule_cache_expiry[cache_key] = current_time + SCHEDULE_CACHE_TTL
                
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

async def check_section_changes(sections: list) -> Tuple[bool, int, int]:
    """Check if the total number of sections has changed."""
    try:
        current_count = len(sections)
        stored_count = await get_section_count_from_redis() or 0
        
        # Only update stored count if we have sections
        if current_count > 0:
            has_changed = current_count != stored_count
            if has_changed:
                logger.info(f"Section count changed: {stored_count} -> {current_count}")
                await save_section_count_to_redis(current_count)
            return has_changed, current_count, stored_count
        return False, stored_count, stored_count
    except Exception as e:
        logger.error(f"Error checking section changes: {str(e)}")
        return False, 0, 0

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
        session_id = request.session.get("id")
        if not session_id:
            return JSONResponse({"error": "No session found"}, status_code=401)
        
        # Get a valid token
        tokens = await load_tokens_from_redis(session_id)
        if not tokens or "access_token" not in tokens or is_token_expired(tokens):
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
                
                # Save final results to Redis
                if new_labs > 0:
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

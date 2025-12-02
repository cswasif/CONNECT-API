#!/usr/bin/env python3
"""
Test script to verify Redis connection and token storage
"""
import os
import json
import time
from main import get_redis_sync, redis_keys_sync, redis_get_sync

def test_redis_connection():
    """Test Redis connection and list existing tokens"""
    print("Testing Redis connection...")
    
    try:
        redis_conn = get_redis_sync()
        print("✓ Redis connection established")
        
        # List all token keys
        token_keys = redis_keys_sync(redis_conn, "tokens:*")
        print(f"Found {len(token_keys)} token keys:")
        
        for key in token_keys:
            data = redis_get_sync(redis_conn, key)
            if data:
                tokens = json.loads(data)
                print(f"  {key}: expires_at={tokens.get('expires_at', 'N/A')}")
                
                # Check if token is expired
                now = int(time.time())
                expires_at = tokens.get('expires_at', 0)
                if expires_at > now:
                    remaining = expires_at - now
                    print(f"    ✓ Active (expires in {remaining}s)")
                else:
                    remaining = now - expires_at
                    print(f"    ✗ Expired ({remaining}s ago)")
        
        return True
        
    except Exception as e:
        print(f"✗ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    test_redis_connection()
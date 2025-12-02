#!/usr/bin/env python3
"""Test Redis connection with the provided URL."""

import os
import sys
import json
from datetime import datetime, timezone

# Add the current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from upstash_redis import Redis as UpstashRedis
    import redis
except ImportError:
    print("❌ Required packages not installed. Run: pip install upstash-redis redis")
    sys.exit(1)

def test_connection():
    """Test Redis connection."""
    
    # Use the provided URL
    redis_url = "rediss://default:@willing-husky-43244.upstash.io:6379"
    
    print(f"Testing Redis URL: {redis_url}")
    
    try:
        # Test Upstash Redis
        client = UpstashRedis(url=redis_url)
        result = client.ping()
        print(f"✅ Redis connection successful: {result}")
        
        # Check for tokens
        keys = client.keys("student_tokens:*")
        print(f"📊 Found {len(keys)} token keys:")
        
        for key in keys:
            try:
                token_data = client.get(key)
                if token_data:
                    data = json.loads(token_data)
                    student_id = key.split(":")[1]
                    expires_at = data.get('expires_at', 0)
                    
                    # Check expiration
                    now = datetime.now(timezone.utc).timestamp()
                    remaining = expires_at - now
                    
                    if remaining > 0:
                        print(f"  ✅ Token for {student_id}: Active ({int(remaining)}s remaining)")
                    else:
                        print(f"  ❌ Token for {student_id}: Expired ({int(remaining)}s ago)")
                        
            except Exception as e:
                print(f"  ⚠️  Error reading token {key}: {e}")
                
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 This might be due to missing token in URL")
        return False

if __name__ == "__main__":
    test_connection()
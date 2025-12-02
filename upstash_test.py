#!/usr/bin/env python3
"""Test Upstash Redis connection with proper format."""

import os
import sys
import json
from datetime import datetime, timezone

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from upstash_redis import Redis
except ImportError:
    print("❌ upstash-redis not installed. Run: pip install upstash-redis")
    sys.exit(1)

def test_upstash_connection():
    """Test Upstash Redis connection with proper token."""
    
    # Extract token from the provided URL
    token = "AajsAAIncDFjY2Q4YWQzMjYwODY0ZWJkYTZjYzYyNGNhNjUwODkzZHAxNDMyNDQ"
    url = "https://willing-husky-43244.upstash.io"
    
    print(f"🔍 Testing Upstash Redis connection...")
    print(f"🔗 URL: {url}")
    print(f"🔑 Token: {token[:10]}...")
    
    try:
        # Create Redis client
        client = Redis(url=url, token=token)
        
        # Test connection
        result = client.ping()
        print(f"✅ Upstash Redis connection successful: {result}")
        
        # Check for tokens
        keys = client.keys("student_tokens:*")
        print(f"📊 Found {len(keys)} token keys:")
        
        active_tokens = 0
        expired_tokens = 0
        
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
                        active_tokens += 1
                    else:
                        print(f"  ❌ Token for {student_id}: Expired ({int(remaining)}s ago)")
                        expired_tokens += 1
                        
            except Exception as e:
                print(f"  ⚠️  Error reading token {key}: {e}")
        
        print(f"\n📈 Summary: {active_tokens} active, {expired_tokens} expired")
        
        if active_tokens > 0:
            print("🎉 Your Redis configuration is working correctly!")
            print("🚀 Next step: Set this configuration in Vercel")
            print("   REDIS_URL=https://willing-husky-43244.upstash.io")
            print("   UPSTASH_REDIS_TOKEN=AajsAAIncDFjY2Q4YWQzMjYwODY0ZWJkYTZjYzYyNGNhNjUwODkzZHAxNDMyNDQ")
        else:
            print("💡 No tokens found - this is normal for a fresh Redis instance")
            print("💡 Add tokens via /enter-tokens after configuring Vercel")
            
        return True
        
    except Exception as e:
        print(f"❌ Upstash Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Upstash Redis Connection")
    print("=" * 50)
    test_upstash_connection()
#!/usr/bin/env python3
"""
Redis Configuration Verification Script

This script helps verify that your Redis configuration is working correctly
for both local development and Vercel deployment.
"""

import os
import redis
import json
from datetime import datetime, timezone
from upstash_redis import Redis as UpstashRedis

def test_redis_connection():
    """Test Redis connection using environment variables."""
    print("🔍 Testing Redis Configuration...")
    
    # Get Redis URL from environment
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        print("❌ REDIS_URL not found in environment variables")
        print("💡 Please set REDIS_URL in your .env file or Vercel environment variables")
        return False
    
    print(f"✅ Found REDIS_URL: {redis_url[:50]}...")
    
    try:
        # Try Upstash Redis first
        if "upstash.io" in redis_url:
            print("🔄 Testing Upstash Redis connection...")
            client = UpstashRedis(url=redis_url)
            client.ping()
            print("✅ Upstash Redis connection successful")
        else:
            # Fallback to redis-py
            print("🔄 Testing redis-py connection...")
            client = redis.from_url(redis_url)
            client.ping()
            print("✅ redis-py connection successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def check_stored_tokens():
    """Check if tokens are properly stored in Redis."""
    print("\n🔍 Checking stored tokens...")
    
    redis_url = os.environ.get("REDIS_URL")
    if not redis_url:
        return
    
    try:
        if "upstash.io" in redis_url:
            client = UpstashRedis(url=redis_url)
        else:
            client = redis.from_url(redis_url)
            
        # Find all token keys
        keys = client.keys("student_tokens:*")
        print(f"📊 Found {len(keys)} token keys")
        
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
                        print(f"✅ Token for {student_id}: Active ({int(remaining)}s remaining)")
                    else:
                        print(f"❌ Token for {student_id}: Expired ({int(remaining)}s ago)")
                        
            except Exception as e:
                print(f"⚠️  Error reading token {key}: {e}")
                
    except Exception as e:
        print(f"❌ Error checking tokens: {e}")

def provide_setup_instructions():
    """Provide setup instructions."""
    print("\n📋 Setup Instructions:")
    print("1. Get your Upstash Redis URL from https://console.upstash.com/redis")
    print("2. Set REDIS_URL environment variable:")
    print("   - Local: Add to .env file")
    print("   - Vercel: Set in Settings → Environment Variables")
    print("3. Redeploy your application")
    print("4. Add tokens via /enter-tokens endpoint")
    print("5. Verify with /global-token-status")

if __name__ == "__main__":
    print("🚀 Redis Configuration Verification")
    print("=" * 50)
    
    # Test connection
    if test_redis_connection():
        check_stored_tokens()
    
    provide_setup_instructions()
    print("\n📖 See TROUBLESHOOTING.md for detailed instructions")
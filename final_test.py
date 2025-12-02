#!/usr/bin/env python3
"""Final Redis connection test with environment variables."""

import os
import sys
import json
from datetime import datetime, timezone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

try:
    from upstash_redis import Redis as UpstashRedis
    import redis
except ImportError:
    print("❌ Required packages not installed. Run: pip install upstash-redis redis python-dotenv")
    sys.exit(1)

def test_connection():
    """Test Redis connection with environment variables."""
    
    # Get Redis URL from environment
    redis_url = os.environ.get("REDIS_URL")
    
    if not redis_url:
        print("❌ REDIS_URL not found in environment variables")
        print("💡 Please check your .env file")
        return False
    
    print(f"🔍 Testing Redis URL: {redis_url}")
    
    try:
        # Test connection
        client = UpstashRedis(url=redis_url)
        result = client.ping()
        print(f"✅ Redis connection successful: {result}")
        
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
            print("🚀 Next step: Redeploy your Vercel app with this REDIS_URL")
        else:
            print("💡 Add tokens via /enter-tokens after redeploy")
            
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Final Redis Configuration Test")
    print("=" * 50)
    test_connection()
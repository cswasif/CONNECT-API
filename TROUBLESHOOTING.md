# 🔧 Redis Configuration Troubleshooting Guide

## Problem: "No active global tokens: 2 expired sessions"

### Root Cause
The deployed Vercel application is using a different Redis instance than your local environment. Tokens saved locally are not visible to the deployed app.

### Solution Steps

#### 1. Get Your Upstash Redis URL

**Option A: From Upstash Console**
1. Go to https://console.upstash.com/redis
2. Select your Redis instance
3. Copy the **REST URL** (starts with `https://`) or **Redis URL** (starts with `rediss://`)

**Option B: Create New Redis Instance**
1. Visit https://console.upstash.com/redis
2. Click "Create Database"
3. Choose region: Singapore (sin1) for best performance
4. Copy the connection string

#### 2. Configure Vercel Environment Variables

**Method A: Vercel Dashboard (Recommended)**
1. Go to https://vercel.com/dashboard
2. Select your project
3. Navigate to Settings → Environment Variables
4. Add new variable:
   - **Name**: `REDIS_URL`
   - **Value**: Your Upstash Redis URL (e.g., `rediss://default:password@host.upstash.io:6379`)
   - **Environment**: Production

**Method B: Vercel CLI**
```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Add environment variable
vercel env add REDIS_URL production

# Deploy with new variables
vercel --prod
```

#### 3. Verify Configuration

**Check Environment Variables:**
```bash
# Using Vercel CLI
vercel env ls

# Or check in dashboard
```

**Test Token Storage:**
1. Visit your deployed app: `https://your-app.vercel.app/`
2. Go to `/enter-tokens`
3. Add new tokens
4. Check `/global-token-status` should show active tokens

#### 4. Quick Fix Commands

**Test Redis Connection:**
```python
# Run this locally to verify Redis connection
python test_redis.py
```

**Force Redeploy:**
```bash
# Trigger fresh deployment
vercel --prod --force
```

### Common Mistakes to Avoid

1. **Using localhost Redis**: Local Redis won't work in production
2. **Wrong URL format**: Must be `rediss://` (with 's' for SSL)
3. **Missing environment variables**: Ensure REDIS_URL is set in production
4. **Token format issues**: Ensure tokens are valid JWT tokens

### Verification Checklist

- [ ] REDIS_URL environment variable set in Vercel
- [ ] Using Upstash Redis (not localhost)
- [ ] URL format: `rediss://default:password@host.upstash.io:6379`
- [ ] New deployment triggered after setting variables
- [ ] Tokens added via deployed web interface
- [ ] `/global-token-status` shows active tokens

### Still Having Issues?

1. **Check logs**: Vercel → Project → Functions → View Logs
2. **Debug mode**: Add `DEBUG_MODE=true` to environment variables
3. **Redis test**: Run `python test_redis.py` locally
4. **Manual verification**: Use Redis CLI to check stored tokens

### Redis CLI Verification

```bash
# Install redis-cli
# Connect to your Upstash Redis
redis-cli -u rediss://default:password@host.upstash.io:6379

# Check for tokens
KEYS student_tokens:*
GET student_tokens:your_student_id
```

### Support Resources

- [Upstash Redis Docs](https://docs.upstash.com/redis)
- [Vercel Environment Variables](https://vercel.com/docs/environment-variables)
- [Redis URL Format](https://github.com/redis/redis-py#connecting-to-redis)
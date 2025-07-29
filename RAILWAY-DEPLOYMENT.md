# 🚀 Railway Deployment Guide

This guide will help you deploy the OSS AI Models API to Railway, a modern hosting platform with zero-configuration deployments.

## 📋 Prerequisites

1. **Railway Account**: [Sign up at railway.com](https://railway.com)
2. **Git Repository**: Your code should be in a Git repository (GitHub, GitLab, etc.)
3. **Railway CLI** (optional): `npm install -g @railway/cli`

## 🚀 Quick Deploy (Recommended)

### Method 1: Deploy from GitHub (Easiest)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Railway deployment"
   git push origin main
   ```

2. **Deploy on Railway**:
   - Go to [railway.com](https://railway.com)
   - Click "Start a New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository
   - Railway will automatically detect it's a Node.js project

3. **Configuration** (Automatic):
   - Health check endpoint: `/api/health`
   - Start command: Auto-detected
   - Build command: Auto-detected

### Method 2: Railway CLI

1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Deploy**:
   ```bash
   railway login
   railway init
   railway up
   ```

## ⚙️ Environment Variables

Railway will automatically set `PORT` and `NODE_ENV=production`. You can add custom variables:

### Required Variables (Optional)
```bash
NODE_ENV=production                    # Auto-set by Railway
PORT=3000                             # Auto-set by Railway
```

### Optional Variables
```bash
# Custom CORS origins (comma-separated)
ALLOWED_ORIGINS=https://your-domain.com,https://api.yourdomain.com

# Rate limiting (defaults provided)
API_RATE_LIMIT_WINDOW=900000          # 15 minutes in ms
API_RATE_LIMIT_MAX=1000               # Max requests per window
```

### Setting Environment Variables

**Via Railway Dashboard**:
1. Go to your project dashboard
2. Click "Variables" tab
3. Add variables as needed

**Via Railway CLI**:
```bash
railway variables set ALLOWED_ORIGINS=https://yourdomain.com
```

## 🌐 Custom Domain Setup

### Using Railway Subdomain
Your app will be available at: `https://your-project-name.up.railway.app`

### Using Custom Domain
1. Go to Railway Dashboard → Settings → Domains
2. Add your custom domain
3. Update DNS records as instructed
4. Update CORS settings in environment variables

## 📊 Post-Deployment Verification

After deployment, verify your API is working:

### 1. Health Check
```bash
curl https://your-app.up.railway.app/api/health
```

Expected response:
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00.000Z",
    "version": "1.0.0",
    "totalModels": 50
  }
}
```

### 2. API Documentation
Visit: `https://your-app.up.railway.app/api`

### 3. Test Endpoints
```bash
# Get first 3 models
curl "https://your-app.up.railway.app/api/models?limit=3"

# Search for language models
curl "https://your-app.up.railway.app/api/models?category=language&limit=2"

# Get statistics
curl "https://your-app.up.railway.app/api/stats"
```

## 🔧 Railway Configuration Files

### `package.json` Scripts
```json
{
  "scripts": {
    "start": "node server.js",
    "railway:start": "node server.js",
    "test": "node test-import.js"
  }
}
```

## 📈 Monitoring & Logging

### Railway Dashboard
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Deployments**: History and rollback options

### Health Monitoring
Railway automatically monitors `/api/health` endpoint:
- **Healthy**: Returns 200 status
- **Unhealthy**: Automatic restart after failures

### Application Logs
View logs in Railway dashboard or via CLI:
```bash
railway logs
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Build Failures
```bash
# Check build logs in Railway dashboard
# Ensure all dependencies are in package.json
npm install --save express cors helmet compression express-rate-limit
```

#### 2. CORS Issues
```bash
# Update allowed origins
railway variables set ALLOWED_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
```

#### 3. Database Not Loading
```bash
# Check if data/models.js exists and is properly formatted
# Verify import in server.js
node test-import.js
```

#### 4. Port Issues
Railway automatically assigns PORT. Don't hardcode it:
```javascript
const PORT = process.env.PORT || 3000; // ✅ Correct
const PORT = 3000; // ❌ Wrong for Railway
```

### Debug Commands
```bash
# Check Railway status
railway status

# View logs
railway logs --tail

# Connect to service shell
railway shell
```

## 🔄 Continuous Deployment

Railway automatically redeploys when you push to your connected Git branch:

```bash
git add .
git commit -m "Update API endpoints"
git push origin main
# Railway automatically deploys 🚀
```

### Manual Deployment
```bash
railway up
```

## 💰 Pricing & Resources

### Railway Free Tier
- **$5 credit/month**
- **512MB RAM**
- **1GB Storage**
- **100GB Bandwidth**

Perfect for the AI Models API which is lightweight and stateless.

### Resource Usage
The AI Models API is very efficient:
- **Memory**: ~50-100MB
- **CPU**: Low usage
- **Storage**: ~5MB (just code + models data)

## 🔐 Security Best Practices

### Environment Variables
- Never commit `.env` files
- Use Railway's secure variable storage
- Rotate any exposed credentials


## 📚 Additional Resources

- **Railway Docs**: [docs.railway.com](https://docs.railway.com)
- **Express on Railway**: [Railway Express Guide](https://docs.railway.com/guides/express)
- **Railway CLI**: [CLI Documentation](https://docs.railway.com/guides/cli)

## 🎉 Success!

After deployment, your API will be available at:
- **API Base**: `https://your-app.up.railway.app/api`
- **Documentation**: `https://your-app.up.railway.app/api`
- **Health Check**: `https://your-app.up.railway.app/api/health`
- **Web Interface**: `https://your-app.up.railway.app`

Your OSS AI Models API is now live and ready to serve users worldwide! 🌍

## 🆘 Support

If you encounter issues:
1. Check Railway dashboard logs
2. Verify environment variables
3. Test health endpoint
4. Review this deployment guide
5. Check Railway's status page: [status.railway.com](https://status.railway.com)
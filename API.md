# OSS AI Models API Documentation

Welcome to the Open Source AI Models Database API! This API provides access to a comprehensive database of 50+ popular open source AI models across various categories.

## Base URL
```
https://ossmodels.dev/api
```

For local development:
```
http://localhost:3000/api
```

## Authentication
No authentication required. The API is free and open for public use.

## Rate Limiting
- **Limit**: 1000 requests per 15-minute window per IP
- **Headers**: Rate limit information is included in response headers

## Response Format
All API responses follow this format:
```json
{
  "success": true,
  "data": { /* response data */ },
  "error": "error message (if success is false)"
}
```

---

## Endpoints

### 1. Health Check
**GET** `/api/health`

Check API status and get basic information.

**Response:**
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

---

### 2. Get All Models
**GET** `/api/models`

Retrieve all AI models with optional filtering, searching, sorting, and pagination.

**Query Parameters:**
| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `category` | string | Filter by category | `language`, `vision`, `audio`, `multimodal`, `embedding` |
| `provider` | string | Filter by provider | `Google`, `Meta`, `OpenAI` |
| `license` | string | Filter by license | `MIT`, `Apache-2.0` |
| `search` | string | Search in name, description, tags | `transformer` |
| `inputModality` | string | Filter by input type | `Text`, `Image`, `Audio` |
| `outputModality` | string | Filter by output type | `Text`, `Image`, `Audio` |
| `toolCalling` | string | Filter by tool calling support | `Yes`, `No`, `Limited` |
| `reasoning` | string | Filter by reasoning capability | `Strong`, `Good`, `Basic`, `None` |
| `sortBy` | string | Sort field | `name`, `date`, `downloads`, `stars`, `parameters` |
| `order` | string | Sort order | `asc`, `desc` |
| `limit` | number | Items per page (max 100) | `20` |
| `offset` | number | Pagination offset | `0` |

**Example Request:**
```bash
GET /api/models?category=language&provider=Meta&limit=10&sortBy=stars&order=desc
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "name": "LLaMA 2",
      "author": "Meta",
      "provider": "Hugging Face",
      "category": "language",
      "description": "A collection of pretrained and fine-tuned large language models...",
      "plainDescription": "A powerful AI that can have conversations...",
      "useCases": ["Chatbots", "Content Writing", "Code Assistance"],
      "inputModalities": ["Text"],
      "outputModalities": ["Text"],
      "technicalSpecs": {
        "parameters": "7B - 70B",
        "memoryRequired": "14GB - 140GB",
        "hardwareRequirement": "GPU recommended",
        "inferenceSpeed": "Fast",
        "formats": ["PyTorch", "GGUF", "AWQ"],
        "toolCalling": "Limited",
        "reasoning": "Good",
        "inputCost": "Free",
        "outputCost": "Free"
      },
      "license": "Custom",
      "downloads": "2.5M",
      "stars": "45.2K",
      "date": "2023-07-18",
      "tags": ["transformer", "instruction-following", "chat"],
      "githubUrl": "https://github.com/facebookresearch/llama",
      "huggingtfaceUrl": "https://huggingface.co/meta-llama"
    }
  ],
  "pagination": {
    "total": 50,
    "limit": 10,
    "offset": 0,
    "hasMore": true
  },
  "filters": {
    "category": "language",
    "provider": "Meta",
    "sortBy": "stars",
    "order": "desc"
  }
}
```

---

### 3. Get Specific Model
**GET** `/api/models/{identifier}`

Retrieve a specific model by name or array index.

**Parameters:**
- `identifier`: Model name (case-insensitive) or array index

**Example Requests:**
```bash
GET /api/models/llama-2
GET /api/models/0
```

**Response:**
```json
{
  "success": true,
  "data": {
    "name": "LLaMA 2",
    "author": "Meta",
    // ... full model data
  }
}
```

---

### 4. Get Filter Options
**GET** `/api/filters`

Get all available filter options for the database.

**Response:**
```json
{
  "success": true,
  "data": {
    "categories": ["language", "vision", "audio", "multimodal", "embedding"],
    "providers": ["Google", "Meta", "OpenAI", "Hugging Face", "Microsoft"],
    "licenses": ["MIT", "Apache-2.0", "GPL-3.0", "Custom"],
    "inputModalities": ["Text", "Image", "Audio", "Video"],
    "outputModalities": ["Text", "Image", "Audio", "Embeddings", "Masks"],
    "toolCalling": ["Yes", "No", "Limited"],
    "reasoning": ["Strong", "Good", "Basic", "None"]
  }
}
```

---

### 5. Get Statistics
**GET** `/api/stats`

Get database statistics and insights.

**Response:**
```json
{
  "success": true,
  "data": {
    "totalModels": 50,
    "categories": {
      "language": 17,
      "vision": 12,
      "audio": 9,
      "multimodal": 8,
      "embedding": 4
    },
    "providers": {
      "Google": 8,
      "Meta": 7,
      "Hugging Face": 15
    },
    "licenses": {
      "MIT": 12,
      "Apache-2.0": 18,
      "GPL-3.0": 5
    },
    "totalDownloads": 50000000,
    "totalStars": 500000
  }
}
```

---

### 6. Vercel AI SDK Compatible Endpoint
**POST** `/api/ai/models`

Optimized endpoint for Vercel AI SDK integration.

**Request Body:**
```json
{
  "query": "language model for coding",
  "filters": {
    "category": "language",
    "toolCalling": "Yes"
  },
  "limit": 5
}
```

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "id": "llama-2",
      "name": "LLaMA 2",
      "description": "A powerful AI that can have conversations...",
      "provider": "Hugging Face",
      "category": "language",
      "capabilities": {
        "toolCalling": true,
        "reasoning": true,
        "inputModalities": ["Text"],
        "outputModalities": ["Text"]
      },
      "pricing": {
        "input": "Free",
        "output": "Free"
      },
      "specs": {
        "parameters": "7B - 70B",
        "memory": "14GB - 140GB",
        "speed": "Fast"
      }
    }
  ],
  "total": 5,
  "query": "language model for coding",
  "filters": {
    "category": "language",
    "toolCalling": "Yes"
  }
}
```

---

## Error Responses

### 404 Not Found
```json
{
  "success": false,
  "error": "Not found",
  "message": "API endpoint /api/invalid not found"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal server error",
  "message": "Something went wrong"
}
```

### 429 Too Many Requests
```json
{
  "success": false,
  "error": "Too many requests from this IP, please try again later.",
  "retryAfter": "15 minutes"
}
```

---

## Usage Examples

### JavaScript/Node.js
```javascript
// Get all language models
const response = await fetch('https://ossmodels.dev/api/models?category=language');
const data = await response.json();
console.log(data.data); // Array of language models

// Search for specific models
const searchResponse = await fetch('https://ossmodels.dev/api/models?search=transformer&limit=5');
const searchData = await searchResponse.json();

// Get a specific model
const modelResponse = await fetch('https://ossmodels.dev/api/models/llama-2');
const model = await modelResponse.json();
```

### Python
```python
import requests

# Get all models with filtering
response = requests.get('https://ossmodels.dev/api/models', params={
    'category': 'vision',
    'provider': 'Google',
    'limit': 10
})
data = response.json()

# Search for models
search_response = requests.get('https://ossmodels.dev/api/models', params={
    'search': 'image generation',
    'outputModality': 'Image'
})
```

### cURL
```bash
# Get all models
curl "https://ossmodels.dev/api/models"

# Filter by category and provider
curl "https://ossmodels.dev/api/models?category=language&provider=OpenAI"

# Get specific model
curl "https://ossmodels.dev/api/models/whisper"

# Get statistics
curl "https://ossmodels.dev/api/stats"
```

---

## Integration with Vercel AI SDK

```typescript
import { createOpenAI } from '@ai-sdk/openai';

// Use the models endpoint to discover available models
async function getAvailableModels() {
  const response = await fetch('https://ossmodels.dev/api/ai/models', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      query: 'coding assistant',
      filters: {
        category: 'language',
        toolCalling: 'Yes'
      },
      limit: 10
    })
  });
  
  const { models } = await response.json();
  return models;
}

// Example usage
const models = await getAvailableModels();
console.log('Available coding models:', models);
```

---

## CORS Policy

The API supports CORS for browser-based applications:

- **Production**: `https://ossmodels.dev`, `https://www.ossmodels.dev`
- **Development**: All origins allowed
- **Methods**: GET, POST, OPTIONS
- **Headers**: Content-Type, Authorization, X-Requested-With

---

## Deployment

### Vercel Deployment
```bash
npm install
vercel --prod
```

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install --production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

### Environment Variables
```env
NODE_ENV=production
PORT=3000
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/oss-ai-models/issues)
- **API Status**: [https://ossmodels.dev/api/health](https://ossmodels.dev/api/health)
- **Documentation**: [https://ossmodels.dev/docs](https://ossmodels.dev/docs)

---

## License

This API and database are open source under the MIT License.
# 🤖 Open Source AI Models Database & API

A comprehensive database of 50+ popular open source AI models with a powerful REST API and beautiful web interface.

## ✨ Features

- **🗄️ Comprehensive Database**: 50+ AI models across language, vision, audio, multimodal, and embedding categories
- **🔍 Advanced Filtering**: Search by category, provider, capabilities, modalities, and more
- **📊 Rich Metadata**: Technical specs, pricing, use cases, and compatibility information
- **🌐 REST API**: Full-featured API with filtering, pagination, and search
- **⚡ Vercel AI SDK Integration**: Ready-to-use endpoints for AI applications
- **📱 Responsive UI**: Beautiful web interface that works on all devices
- **🚀 Zero Setup**: Deploy to Vercel with one click

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd aiGPT
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Web Interface: http://localhost:3000
   - API: http://localhost:3000/api
   - API Health: http://localhost:3000/api/health

### Production Deployment

#### Deploy to Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

#### Deploy with Docker
```bash
# Build the image
docker build -t oss-ai-models .

# Run the container
docker run -p 3000:3000 oss-ai-models
```

## 📖 API Usage

### Base URL
```
https://ossmodels.dev/api
```

### Get All Models
```bash
curl "https://ossmodels.dev/api/models"
```

### Filter Models
```bash
# Get language models from Google
curl "https://ossmodels.dev/api/models?category=language&provider=Google"

# Search for image generation models
curl "https://ossmodels.dev/api/models?search=image&outputModality=Image"

# Get models with tool calling support
curl "https://ossmodels.dev/api/models?toolCalling=Yes&limit=10"
```

### Get Specific Model
```bash
curl "https://ossmodels.dev/api/models/llama-2"
```

### Get Statistics
```bash
curl "https://ossmodels.dev/api/stats"
```

## 🔧 Integration Examples

### JavaScript/Node.js
```javascript
// Fetch all language models
const response = await fetch('https://ossmodels.dev/api/models?category=language');
const { data } = await response.json();

// Search for specific capabilities
const codingModels = await fetch('https://ossmodels.dev/api/models?search=coding&toolCalling=Yes');
const results = await codingModels.json();
```

### Python
```python
import requests

# Get vision models
response = requests.get('https://ossmodels.dev/api/models', params={
    'category': 'vision',
    'provider': 'Meta',
    'limit': 5
})
models = response.json()['data']
```

### Vercel AI SDK
```typescript
// Use with Vercel AI SDK for model discovery
async function findBestModel(requirements: string) {
  const response = await fetch('https://ossmodels.dev/api/ai/models', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: requirements,
      filters: { toolCalling: 'Yes' },
      limit: 5
    })
  });
  
  const { models } = await response.json();
  return models[0]; // Return the best match
}
```

## 🏗️ Project Structure

```
├── data/
│   └── models.js          # Models database
├── server.js              # Express.js API server
├── index.html             # Frontend interface
├── styles.css             # Styling
├── script.js              # Frontend JavaScript
├── package.json           # Dependencies
├── vercel.json            # Vercel deployment config
├── API.md                 # Full API documentation
└── README.md              # This file
```

## 📊 Database Schema

Each model includes:

```typescript
interface AIModel {
  name: string;
  author: string;
  provider: string;
  category: 'language' | 'vision' | 'audio' | 'multimodal' | 'embedding';
  description: string;
  plainDescription: string;
  useCases: string[];
  inputModalities: string[];
  outputModalities: string[];
  technicalSpecs: {
    parameters: string;
    memoryRequired: string;
    hardwareRequirement: string;
    inferenceSpeed: string;
    formats: string[];
    toolCalling: 'Yes' | 'No' | 'Limited';
    reasoning: 'Strong' | 'Good' | 'Basic' | 'None';
    inputCost: string;
    outputCost: string;
  };
  license: string;
  downloads: string;
  stars: string;
  date: string;
  tags: string[];
  githubUrl: string;
  huggingtfaceUrl: string;
}
```

## 🌟 Featured Models

### Language Models
- **LLaMA 2** - Meta's powerful conversational AI
- **GPT-J** - EleutherAI's open alternative to GPT-3
- **T5** - Google's versatile text-to-text transformer
- **Falcon** - Technology Innovation Institute's efficient LLM

### Vision Models
- **Stable Diffusion XL** - High-quality text-to-image generation
- **YOLOv8** - Real-time object detection
- **SAM** - Segment anything model
- **Vision Transformer (ViT)** - Revolutionary image classification

### Audio Models
- **Whisper** - OpenAI's speech recognition
- **MusicGen** - Meta's music generation
- **Bark** - Suno AI's creative text-to-audio

### Multimodal Models
- **CLIP** - OpenAI's vision-language model
- **LLaVA** - Large language and vision assistant
- **BLIP-2** - Salesforce's advanced multimodal AI

## 🔍 Search & Filter Options

- **Categories**: language, vision, audio, multimodal, embedding
- **Providers**: Google, Meta, OpenAI, Microsoft, Hugging Face, and 20+ more
- **Capabilities**: Tool calling, reasoning levels, modalities
- **Technical**: Parameters, memory requirements, inference speed
- **Licensing**: MIT, Apache-2.0, GPL-3.0, Custom
- **Free Text Search**: Names, descriptions, tags, use cases

## 🛠️ API Features

- **RESTful Design**: Clean, predictable endpoints
- **Advanced Filtering**: Multiple filter combinations
- **Pagination**: Efficient data loading
- **Search**: Full-text search across all fields
- **Sorting**: Multiple sort options
- **Rate Limiting**: 1000 requests per 15 minutes
- **CORS Enabled**: Works with browser applications
- **Error Handling**: Comprehensive error responses
- **Statistics**: Database insights and metrics

## 📈 Performance

- **Response Time**: < 100ms for most queries
- **Database Size**: 50 models, ~2MB total
- **Caching**: Intelligent caching for optimal performance
- **CDN Ready**: Optimized for global distribution

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Add Models**: Submit new open source AI models
2. **Improve Data**: Enhance model descriptions and metadata
3. **Fix Bugs**: Report and fix issues
4. **Documentation**: Improve guides and examples

## 📝 License

MIT License - feel free to use this in your own projects!

## 🔗 Links

- **API Documentation**: [API.md](./API.md)
- **Live Demo**: https://ossmodels.dev
- **API Health**: https://ossmodels.dev/api/health
- **Statistics**: https://ossmodels.dev/api/stats

## 📧 Support

- **Issues**: GitHub Issues
- **API Status**: Check `/api/health` endpoint
- **Rate Limits**: Monitor response headers

---

Built with ❤️ for the open source AI community
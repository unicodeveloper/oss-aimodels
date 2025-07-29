const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const path = require('path');
const modelsDatabase = require('./data/models');

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(compression());

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.API_RATE_LIMIT_WINDOW) || 15 * 60 * 1000, // 15 minutes
  max: parseInt(process.env.API_RATE_LIMIT_MAX) || 1000, // limit each IP to 1000 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes'
  },
  standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false, // Disable the `X-RateLimit-*` headers
});
app.use('/api/', limiter);

// CORS configuration
const getAllowedOrigins = () => {
  if (process.env.ALLOWED_ORIGINS) {
    return process.env.ALLOWED_ORIGINS.split(',').map(origin => origin.trim());
  }
  
  if (process.env.NODE_ENV === 'production') {
    return [
      /^https:\/\/.*\.railway\.app$/,
      /^https:\/\/.*\.up\.railway\.app$/
    ];
  }
  
  return true; // Allow all origins in development
};

app.use(cors({
  origin: getAllowedOrigins(),
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  credentials: true
}));

// Body parsing middleware
app.use(express.json({ limit: '800mb' }));
app.use(express.urlencoded({ extended: true, limit: '800mb' }));

// Serve static files
app.use(express.static('.', {
  index: 'index.html',
  setHeaders: (res, path) => {
    if (path.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache');
    } else {
      res.setHeader('Cache-Control', 'public, max-age=31536000');
    }
  }
}));

// API Routes

// API Documentation landing page
app.get('/api', (req, res) => {
  res.sendFile(path.join(__dirname, 'api-docs.html'));
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    totalModels: modelsDatabase.length
  });
});

// Get all models with optional filtering
app.get('/api/models', (req, res) => {
  try {
    let filteredModels = [...modelsDatabase];
    
    // Apply filters
    const { 
      category, 
      provider, 
      license, 
      search, 
      limit = 50, 
      offset = 0,
      sortBy = 'name',
      order = 'asc',
      inputModality,
      outputModality,
      toolCalling,
      reasoning
    } = req.query;

    // Category filter
    if (category) {
      filteredModels = filteredModels.filter(model => 
        model.category.toLowerCase() === category.toLowerCase()
      );
    }

    // Provider filter
    if (provider) {
      filteredModels = filteredModels.filter(model => 
        model.provider.toLowerCase() === provider.toLowerCase()
      );
    }

    // License filter
    if (license) {
      filteredModels = filteredModels.filter(model => 
        model.license.toLowerCase() === license.toLowerCase()
      );
    }

    // Input modality filter
    if (inputModality) {
      filteredModels = filteredModels.filter(model => 
        model.inputModalities.some(mod => 
          mod.toLowerCase() === inputModality.toLowerCase()
        )
      );
    }

    // Output modality filter  
    if (outputModality) {
      filteredModels = filteredModels.filter(model => 
        model.outputModalities.some(mod => 
          mod.toLowerCase() === outputModality.toLowerCase()
        )
      );
    }

    // Tool calling filter
    if (toolCalling) {
      filteredModels = filteredModels.filter(model => 
        model.technicalSpecs.toolCalling.toLowerCase() === toolCalling.toLowerCase()
      );
    }

    // Reasoning filter
    if (reasoning) {
      filteredModels = filteredModels.filter(model => 
        model.technicalSpecs.reasoning.toLowerCase() === reasoning.toLowerCase()
      );
    }

    // Search filter
    if (search) {
      const searchTerm = search.toLowerCase();
      filteredModels = filteredModels.filter(model => 
        model.name.toLowerCase().includes(searchTerm) ||
        model.author.toLowerCase().includes(searchTerm) ||
        model.description.toLowerCase().includes(searchTerm) ||
        model.plainDescription.toLowerCase().includes(searchTerm) ||
        model.tags.some(tag => tag.toLowerCase().includes(searchTerm)) ||
        model.useCases.some(useCase => useCase.toLowerCase().includes(searchTerm))
      );
    }

    // Sorting
    filteredModels.sort((a, b) => {
      let aValue, bValue;
      
      switch(sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'date':
          aValue = new Date(a.date);
          bValue = new Date(b.date);
          break;
        case 'downloads':
          aValue = parseFloat(a.downloads);
          bValue = parseFloat(b.downloads);
          break;
        case 'stars':
          aValue = parseFloat(a.stars);
          bValue = parseFloat(b.stars);
          break;
        case 'parameters':
          aValue = parseFloat(a.technicalSpecs.parameters);
          bValue = parseFloat(b.technicalSpecs.parameters);
          break;
        default:
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
      }

      if (order === 'desc') {
        return aValue < bValue ? 1 : aValue > bValue ? -1 : 0;
      } else {
        return aValue > bValue ? 1 : aValue < bValue ? -1 : 0;
      }
    });

    // Pagination
    const total = filteredModels.length;
    const limitNum = Math.min(parseInt(limit), 100); // Max 100 items per request
    const offsetNum = parseInt(offset);
    const paginatedModels = filteredModels.slice(offsetNum, offsetNum + limitNum);

    res.json({
      success: true,
      data: paginatedModels,
      pagination: {
        total,
        limit: limitNum,
        offset: offsetNum,
        hasMore: offsetNum + limitNum < total
      },
      filters: {
        category,
        provider,
        license,
        search,
        inputModality,
        outputModality,
        toolCalling,
        reasoning,
        sortBy,
        order
      }
    });

  } catch (error) {
    console.error('Error in /api/models:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to fetch models'
    });
  }
});

// Get a specific model by name or ID
app.get('/api/models/:identifier', (req, res) => {
  try {
    const { identifier } = req.params;
    
    // Try to find by name (case-insensitive) or by array index
    const model = modelsDatabase.find(m => 
      m.name.toLowerCase() === identifier.toLowerCase()
    ) || modelsDatabase[parseInt(identifier)];

    if (!model) {
      return res.status(404).json({
        success: false,
        error: 'Model not found',
        message: `No model found with identifier: ${identifier}`
      });
    }

    res.json({
      success: true,
      data: model
    });

  } catch (error) {
    console.error('Error in /api/models/:identifier:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to fetch model'
    });
  }
});

// Get available filter options
app.get('/api/filters', (req, res) => {
  try {
    const categories = [...new Set(modelsDatabase.map(m => m.category))];
    const providers = [...new Set(modelsDatabase.map(m => m.provider))];
    const licenses = [...new Set(modelsDatabase.map(m => m.license))];
    const inputModalities = [...new Set(modelsDatabase.flatMap(m => m.inputModalities))];
    const outputModalities = [...new Set(modelsDatabase.flatMap(m => m.outputModalities))];
    const toolCallingOptions = [...new Set(modelsDatabase.map(m => m.technicalSpecs.toolCalling))];
    const reasoningOptions = [...new Set(modelsDatabase.map(m => m.technicalSpecs.reasoning))];

    res.json({
      success: true,
      data: {
        categories: categories.sort(),
        providers: providers.sort(),
        licenses: licenses.sort(),
        inputModalities: inputModalities.sort(),
        outputModalities: outputModalities.sort(),
        toolCalling: toolCallingOptions.sort(),
        reasoning: reasoningOptions.sort()
      }
    });

  } catch (error) {
    console.error('Error in /api/filters:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to fetch filter options'
    });
  }
});

// Get statistics
app.get('/api/stats', (req, res) => {
  try {
    const stats = {
      totalModels: modelsDatabase.length,
      categories: {},
      providers: {},
      licenses: {},
      averageParameters: 0,
      totalDownloads: 0,
      totalStars: 0
    };

    // Calculate category distribution
    modelsDatabase.forEach(model => {
      stats.categories[model.category] = (stats.categories[model.category] || 0) + 1;
      stats.providers[model.provider] = (stats.providers[model.provider] || 0) + 1;
      stats.licenses[model.license] = (stats.licenses[model.license] || 0) + 1;
      
      // Convert download and star counts to numbers
      const downloads = parseFloat(model.downloads.replace(/[KMB]/i, '')) * 
        (model.downloads.includes('K') ? 1000 : 
         model.downloads.includes('M') ? 1000000 : 
         model.downloads.includes('B') ? 1000000000 : 1);
      
      const stars = parseFloat(model.stars.replace(/[KMB]/i, '')) * 
        (model.stars.includes('K') ? 1000 : 
         model.stars.includes('M') ? 1000000 : 
         model.stars.includes('B') ? 1000000000 : 1);
      
      stats.totalDownloads += downloads;
      stats.totalStars += stars;
    });

    res.json({
      success: true,
      data: stats
    });

  } catch (error) {
    console.error('Error in /api/stats:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to fetch statistics'
    });
  }
});

// Vercel AI SDK compatible endpoint
app.post('/api/ai/models', (req, res) => {
  try {
    const { query, filters = {}, limit = 10 } = req.body;
    
    let results = [...modelsDatabase];
    
    // Apply filters
    Object.entries(filters).forEach(([key, value]) => {
      if (value) {
        results = results.filter(model => {
          switch(key) {
            case 'category':
              return model.category.toLowerCase() === value.toLowerCase();
            case 'provider':
              return model.provider.toLowerCase() === value.toLowerCase();
            case 'toolCalling':
              return model.technicalSpecs.toolCalling.toLowerCase() === value.toLowerCase();
            case 'reasoning':
              return model.technicalSpecs.reasoning.toLowerCase() === value.toLowerCase();
            default:
              return true;
          }
        });
      }
    });
    
    // Apply search query if provided
    if (query) {
      const searchTerm = query.toLowerCase();
      results = results.filter(model => 
        model.name.toLowerCase().includes(searchTerm) ||
        model.description.toLowerCase().includes(searchTerm) ||
        model.plainDescription.toLowerCase().includes(searchTerm) ||
        model.useCases.some(useCase => useCase.toLowerCase().includes(searchTerm))
      );
    }
    
    // Limit results
    results = results.slice(0, Math.min(limit, 20));
    
    // Format for Vercel AI SDK
    const formattedResults = results.map(model => ({
      id: model.name.toLowerCase().replace(/\s+/g, '-'),
      name: model.name,
      description: model.plainDescription,
      provider: model.provider,
      category: model.category,
      capabilities: {
        toolCalling: model.technicalSpecs.toolCalling !== 'No',
        reasoning: model.technicalSpecs.reasoning !== 'None',
        inputModalities: model.inputModalities,
        outputModalities: model.outputModalities
      },
      pricing: {
        input: model.technicalSpecs.inputCost,
        output: model.technicalSpecs.outputCost
      },
      specs: {
        parameters: model.technicalSpecs.parameters,
        memory: model.technicalSpecs.memoryRequired,
        speed: model.technicalSpecs.inferenceSpeed
      }
    }));
    
    res.json({
      success: true,
      models: formattedResults,
      total: results.length,
      query,
      filters
    });
    
  } catch (error) {
    console.error('Error in /api/ai/models:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: 'Failed to process AI models request'
    });
  }
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({
    success: false,
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler for API routes
app.use('/api/*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Not found',
    message: `API endpoint ${req.originalUrl} not found`
  });
});

// Serve the frontend for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 OSS AI Models API running on port ${PORT}`);
  console.log(`📊 Database contains ${modelsDatabase.length} models`);
  console.log(`🌍 Environment: ${process.env.NODE_ENV || 'development'}`);
  
  if (process.env.NODE_ENV === 'production') {
    console.log(`🌐 API available at /api`);
    console.log(`📋 API Documentation at /api`);
    console.log(`❤️ Health check at /api/health`);
  } else {
    console.log(`🌐 API available at http://localhost:${PORT}/api`);
    console.log(`🔧 Frontend available at http://localhost:${PORT}`);
    console.log(`📋 API Documentation at http://localhost:${PORT}/api`);
  }
});

module.exports = app;
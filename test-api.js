#!/usr/bin/env node

// Simple API test script
const http = require('http');

const BASE_URL = 'http://localhost:3000';

async function testAPI() {
  console.log('🧪 Testing OSS AI Models API...\n');

  const tests = [
    {
      name: 'Health Check',
      path: '/api/health',
      method: 'GET'
    },
    {
      name: 'Get All Models',
      path: '/api/models?limit=5',
      method: 'GET'
    },
    {
      name: 'Search Models',
      path: '/api/models?search=transformer&limit=3',
      method: 'GET'
    },
    {
      name: 'Filter by Category',
      path: '/api/models?category=language&limit=3',
      method: 'GET'
    },
    {
      name: 'Get Specific Model',
      path: '/api/models/llama-2',
      method: 'GET'
    },
    {
      name: 'Get Statistics',
      path: '/api/stats',
      method: 'GET'
    },
    {
      name: 'Get Filter Options',
      path: '/api/filters',
      method: 'GET'
    }
  ];

  for (const test of tests) {
    try {
      console.log(`🔍 Testing: ${test.name}`);
      
      const response = await makeRequest(test.path, test.method);
      const data = JSON.parse(response);
      
      if (data.success) {
        console.log(`✅ ${test.name}: PASSED`);
        
        // Log some relevant info
        if (test.name === 'Health Check') {
          console.log(`   Status: ${data.data.status}, Models: ${data.data.totalModels}`);
        } else if (test.name === 'Get All Models') {
          console.log(`   Returned: ${data.data.length} models`);
        } else if (test.name === 'Get Statistics') {
          console.log(`   Total Models: ${data.data.totalModels}`);
        }
      } else {
        console.log(`❌ ${test.name}: FAILED`);
        console.log(`   Error: ${data.error}`);
      }
    } catch (error) {
      console.log(`❌ ${test.name}: ERROR`);
      console.log(`   ${error.message}`);
    }
    
    console.log('');
  }

  console.log('🎉 API testing complete!');
}

function makeRequest(path, method = 'GET') {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: 3000,
      path: path,
      method: method,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const req = http.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        if (res.statusCode >= 200 && res.statusCode < 300) {
          resolve(data);
        } else {
          reject(new Error(`HTTP ${res.statusCode}: ${data}`));
        }
      });
    });

    req.on('error', (error) => {
      reject(error);
    });

    req.end();
  });
}

// Check if server is running, then run tests
console.log('🚀 Starting API tests...');
console.log(`📍 Testing against: ${BASE_URL}\n`);

// Give server a moment to start if just launched
setTimeout(() => {
  testAPI().catch(error => {
    console.error('❌ Test suite failed:', error.message);
    console.log('\n💡 Make sure the server is running: npm start');
    process.exit(1);
  });
}, 1000);
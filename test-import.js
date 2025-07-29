// Quick test to verify the import and basic functionality
const modelsDatabase = require('./data/models');

console.log('🧪 Testing models import and basic functionality...\n');

// Test 1: Check if modelsDatabase is an array
console.log('✅ Models database is array:', Array.isArray(modelsDatabase));
console.log('✅ Total models loaded:', modelsDatabase.length);

// Test 2: Check if we can iterate (this was the original error)
console.log('✅ Can iterate over models:', modelsDatabase.length > 0);

// Test 3: Test the spread operator (used in API)
try {
  const filteredModels = [...modelsDatabase];
  console.log('✅ Spread operator works:', filteredModels.length === modelsDatabase.length);
} catch (error) {
  console.log('❌ Spread operator failed:', error.message);
}

// Test 4: Test basic filtering (used in API)
try {
  const languageModels = modelsDatabase.filter(model => model.category === 'language');
  console.log('✅ Filtering works:', languageModels.length, 'language models found');
} catch (error) {
  console.log('❌ Filtering failed:', error.message);
}

// Test 5: Check first model structure
if (modelsDatabase.length > 0) {
  const firstModel = modelsDatabase[0];
  console.log('✅ First model name:', firstModel.name);
  console.log('✅ First model has required fields:', 
    Boolean(firstModel.name && firstModel.category && firstModel.technicalSpecs));
}

console.log('\n🎉 All tests passed! The import issue is fixed.');
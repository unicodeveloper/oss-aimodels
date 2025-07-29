// Frontend JavaScript for OSS AI Models Database
// Data is fetched from API instead of being hardcoded

let modelsDatabase = [];
let filteredModels = [];

// Fetch models from API on page load
async function loadModelsFromAPI() {
    try {
        // Show loading state
        document.getElementById('modelsGrid').innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px;">
                <h3>🔄 Loading AI Models...</h3>
                <p>Fetching the latest model data from our API</p>
            </div>
        `;

        const response = await fetch('/api/models?limit=100');
        const result = await response.json();
        
        if (result.success) {
            modelsDatabase = result.data;
            filteredModels = [...modelsDatabase];
            renderModels(filteredModels);
        } else {
            console.error('Failed to load models from API:', result.error);
            showErrorMessage('Failed to load AI models', 'Please try refreshing the page');
        }
    } catch (error) {
        console.error('Error fetching models:', error);
        showErrorMessage('Connection Error', 'Unable to connect to API. Please check your connection and try again.');
    }
}

function showErrorMessage(title, message) {
    document.getElementById('modelsGrid').innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 40px; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
            <h3 style="color: #e74c3c; margin-bottom: 10px;">⚠️ ${title}</h3>
            <p style="color: #666; margin-bottom: 20px;">${message}</p>
            <button onclick="loadModelsFromAPI()" style="background: #667eea; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer;">
                🔄 Retry
            </button>
        </div>
    `;
}

function renderModels(models) {
    const modelsGrid = document.getElementById('modelsGrid');
    modelsGrid.innerHTML = '';

    if (models.length === 0) {
        modelsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; padding: 40px; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1);">
                <h3 style="color: #667eea; margin-bottom: 10px;">🔍 No Models Found</h3>
                <p style="color: #666;">Try adjusting your search or filter criteria</p>
            </div>
        `;
        return;
    }

    models.forEach(model => {
        const modelCard = createModelCard(model);
        modelsGrid.appendChild(modelCard);
    });
}

function createModelCard(model) {
    const card = document.createElement('div');
    card.className = 'model-card';
    
    const categoryClass = `category-${model.category}`;
    const tagsHtml = model.tags.map(tag => `<span class="tag">${tag}</span>`).join('');
    const useCasesHtml = model.useCases.map(useCase => `<span class="use-case">${useCase}</span>`).join('');
    const inputModalitiesHtml = model.inputModalities.map(modality => `<span class="modality input-modality">${modality}</span>`).join('');
    const outputModalitiesHtml = model.outputModalities.map(modality => `<span class="modality output-modality">${modality}</span>`).join('');
    const techSpecsHtml = `
        <div class="tech-spec"><strong>Size:</strong> ${model.technicalSpecs.parameters}</div>
        <div class="tech-spec"><strong>Memory:</strong> ${model.technicalSpecs.memoryRequired}</div>
        <div class="tech-spec"><strong>Hardware:</strong> ${model.technicalSpecs.hardwareRequirement}</div>
        <div class="tech-spec"><strong>Speed:</strong> ${model.technicalSpecs.inferenceSpeed}</div>
        <div class="tech-spec"><strong>Tool Calling:</strong> ${model.technicalSpecs.toolCalling}</div>
        <div class="tech-spec"><strong>Reasoning:</strong> ${model.technicalSpecs.reasoning}</div>
        <div class="tech-spec"><strong>Input Cost:</strong> ${model.technicalSpecs.inputCost}</div>
        <div class="tech-spec"><strong>Output Cost:</strong> ${model.technicalSpecs.outputCost}</div>
    `;
    
    card.innerHTML = `
        <div class="model-header">
            <div>
                <div class="model-name">${model.name}</div>
                <div class="model-author">by ${model.author}</div>
                <div class="model-provider">via ${model.provider}</div>
            </div>
            <span class="category-badge ${categoryClass}">${model.category}</span>
        </div>
        
        <div class="model-description">${model.description}</div>
        <div class="model-plain-description">${model.plainDescription}</div>
        
        <div class="model-stats">
            <span>⭐ ${model.stars}</span>
            <span>📥 ${model.downloads}</span>
            <span>📅 ${formatDate(model.date)}</span>
        </div>
        
        <div class="modalities-section">
            <div class="modality-group">
                <div class="section-title">📥 Input</div>
                <div class="modalities">${inputModalitiesHtml}</div>
            </div>
            <div class="modality-group">
                <div class="section-title">📤 Output</div>
                <div class="modalities">${outputModalitiesHtml}</div>
            </div>
        </div>
        
        <div class="use-cases-section">
            <div class="section-title">💡 Use Cases</div>
            <div class="use-cases">${useCasesHtml}</div>
        </div>
        
        <div class="tech-specs-section">
            <div class="section-title">⚙️ Technical Specs</div>
            <div class="tech-specs">${techSpecsHtml}</div>
        </div>
        
        <div class="model-tags">${tagsHtml}</div>
        
        <div class="model-footer">
            <span class="license">${model.license}</span>
            <div class="model-links">
                <a href="${model.githubUrl}" class="btn btn-secondary" target="_blank">GitHub</a>
                <a href="${model.huggingtfaceUrl}" class="btn btn-primary" target="_blank">🤗 Hub</a>
            </div>
        </div>
    `;
    
    return card;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric' 
    });
}

function filterModels() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const categoryFilter = document.getElementById('categoryFilter').value;
    const licenseFilter = document.getElementById('licenseFilter').value;
    const providerFilter = document.getElementById('providerFilter').value;
    
    filteredModels = modelsDatabase.filter(model => {
        const matchesSearch = model.name.toLowerCase().includes(searchTerm) ||
                            model.author.toLowerCase().includes(searchTerm) ||
                            model.description.toLowerCase().includes(searchTerm) ||
                            model.provider.toLowerCase().includes(searchTerm) ||
                            model.tags.some(tag => tag.toLowerCase().includes(searchTerm));
        
        const matchesCategory = !categoryFilter || model.category === categoryFilter;
        const matchesLicense = !licenseFilter || model.license === licenseFilter;
        const matchesProvider = !providerFilter || model.provider === providerFilter;
        
        return matchesSearch && matchesCategory && matchesLicense && matchesProvider;
    });
    
    sortModels();
}

function sortModels() {
    const sortBy = document.getElementById('sortBy').value;
    
    filteredModels.sort((a, b) => {
        switch(sortBy) {
            case 'name':
                return a.name.localeCompare(b.name);
            case 'date':
                return new Date(b.date) - new Date(a.date);
            case 'downloads':
                return parseFloat(b.downloads) - parseFloat(a.downloads);
            case 'stars':
                return parseFloat(b.stars) - parseFloat(a.stars);
            default:
                return 0;
        }
    });
    
    renderModels(filteredModels);
}

// Event listeners
document.getElementById('searchInput').addEventListener('input', filterModels);
document.getElementById('categoryFilter').addEventListener('change', filterModels);
document.getElementById('licenseFilter').addEventListener('change', filterModels);
document.getElementById('providerFilter').addEventListener('change', filterModels);
document.getElementById('sortBy').addEventListener('change', sortModels);

// Load models when page loads
document.addEventListener('DOMContentLoaded', loadModelsFromAPI);
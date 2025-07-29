const modelsDatabase = [
    {
        name: "LLaMA 2",
        author: "Meta",
        provider: "Hugging Face",
        category: "language",
        description: "A collection of pretrained and fine-tuned large language models ranging from 7B to 70B parameters.",
        plainDescription: "A powerful AI that can have conversations, answer questions, write content, and help with various text tasks.",
        useCases: ["Chatbots", "Content Writing", "Code Assistance", "Question Answering", "Text Summarization"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 70B",
            memoryRequired: "14GB - 140GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "AWQ"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Custom",
        downloads: "2.5M",
        stars: "45.2K",
        date: "2023-07-18",
        tags: ["transformer", "instruction-following", "chat"],
        githubUrl: "https://github.com/facebookresearch/llama",
        huggingtfaceUrl: "https://huggingface.co/meta-llama"
    },
    {
        name: "Stable Diffusion XL",
        author: "Stability AI",
        provider: "Hugging Face",
        category: "vision",
        description: "Latest text-to-image generation model with improved image quality and composition adherence.",
        plainDescription: "Creates high-quality images from text descriptions. Just type what you want to see and it generates the image.",
        useCases: ["Art Generation", "Marketing Materials", "Concept Art", "Social Media Content", "Product Mockups"],
        inputModalities: ["Text"],
        outputModalities: ["Image"],
        technicalSpecs: {
            parameters: "3.5B",
            memoryRequired: "8GB VRAM",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["Diffusers", "ONNX", "SafeTensors"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "OpenRAIL++",
        downloads: "1.8M",
        stars: "32.1K",
        date: "2023-07-26",
        tags: ["diffusion", "text-to-image", "generative"],
        githubUrl: "https://github.com/Stability-AI/generative-models",
        huggingtfaceUrl: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0"
    },
    {
        name: "Whisper",
        author: "OpenAI",
        provider: "Hugging Face",
        category: "audio",
        description: "Automatic speech recognition system trained on 680,000 hours of multilingual data.",
        plainDescription: "Converts speech to text in 99+ languages. Upload audio files and get accurate transcriptions.",
        useCases: ["Meeting Transcription", "Podcast Subtitles", "Voice Notes", "Language Learning", "Accessibility"],
        inputModalities: ["Audio"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "39M - 1.55B",
            memoryRequired: "1GB - 10GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "TensorFlow", "ONNX", "CoreML"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "950K",
        stars: "58.3K",
        date: "2022-09-21",
        tags: ["speech-recognition", "multilingual", "transcription"],
        githubUrl: "https://github.com/openai/whisper",
        huggingtfaceUrl: "https://huggingface.co/openai/whisper-large-v3"
    },
    {
        name: "CLIP",
        author: "OpenAI",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Connects text and images, learning visual concepts from natural language supervision.",
        plainDescription: "Understands both images and text together. Can search images by description or classify images without training.",
        useCases: ["Image Search", "Content Moderation", "Image Classification", "Visual Q&A", "Recommendation Systems"],
        inputModalities: ["Text", "Image"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "428M",
            memoryRequired: "2GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "ONNX", "TensorFlow"],
            toolCalling: "No",
            reasoning: "Basic",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "1.2M",
        stars: "19.8K",
        date: "2021-01-05",
        tags: ["vision-language", "zero-shot", "contrastive"],
        githubUrl: "https://github.com/openai/CLIP",
        huggingtfaceUrl: "https://huggingface.co/openai/clip-vit-large-patch14"
    },
    {
        name: "Sentence Transformers",
        author: "UKP Lab",
        provider: "Hugging Face",
        category: "embedding",
        description: "Framework for state-of-the-art sentence, text and image embeddings.",
        plainDescription: "Converts text into numerical representations that capture meaning. Finds similar sentences and enables semantic search across documents.",
        useCases: ["Document Search", "Text Similarity", "Clustering", "Recommendation Systems", "FAQ Matching"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "22M - 335M",
            memoryRequired: "1GB - 4GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "TensorFlow", "ONNX", "Sentence-Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "3.1M",
        stars: "13.5K",
        date: "2019-08-27",
        tags: ["embeddings", "similarity", "semantic-search"],
        githubUrl: "https://github.com/UKPLab/sentence-transformers",
        huggingtfaceUrl: "https://huggingface.co/sentence-transformers"
    },
    {
        name: "CodeT5",
        author: "Salesforce",
        provider: "Hugging Face",
        category: "language",
        description: "Unified pre-trained encoder-decoder Transformer for code understanding and generation.",
        plainDescription: "AI assistant for programming tasks. Understands code, generates new code, explains functions, and helps with debugging across multiple programming languages.",
        useCases: ["Code Generation", "Bug Fixing", "Code Documentation", "Code Translation", "API Generation"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "60M - 770M",
            memoryRequired: "2GB - 8GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "TensorFlow", "ONNX"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "450K",
        stars: "2.1K",
        date: "2021-09-02",
        tags: ["code-generation", "programming", "transformer"],
        githubUrl: "https://github.com/salesforce/CodeT5",
        huggingtfaceUrl: "https://huggingface.co/Salesforce/codet5-large"
    },
    {
        name: "YOLOv8",
        author: "Ultralytics",
        provider: "Ultralytics",
        category: "vision",
        description: "State-of-the-art real-time object detection, instance segmentation and image classification.",
        plainDescription: "Detects and identifies objects in images and videos in real-time. Can recognize people, cars, animals, and thousands of other objects instantly.",
        useCases: ["Security Cameras", "Autonomous Vehicles", "Quality Control", "Sports Analytics", "Retail Analytics"],
        inputModalities: ["Image", "Video"],
        outputModalities: ["Annotations"],
        technicalSpecs: {
            parameters: "3M - 68M",
            memoryRequired: "1GB - 4GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "ONNX", "TensorRT", "CoreML", "OpenVINO"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "GPL-3.0",
        downloads: "2.8M",
        stars: "22.4K",
        date: "2023-01-10",
        tags: ["object-detection", "real-time", "computer-vision"],
        githubUrl: "https://github.com/ultralytics/ultralytics",
        huggingtfaceUrl: "https://huggingface.co/spaces/Ultralytics/YOLOv8"
    },
    {
        name: "MusicGen",
        author: "Meta",
        provider: "Hugging Face",
        category: "audio",
        description: "Controllable music generation model that produces high-quality music samples.",
        plainDescription: "Creates original music from text descriptions. Specify genre, mood, instruments, and style to generate custom soundtracks and melodies.",
        useCases: ["Content Creation", "Game Soundtracks", "Advertisement Music", "Podcast Intros", "Creative Composition"],
        inputModalities: ["Text"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "300M - 3.3B",
            memoryRequired: "4GB - 16GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "CC-BY-SA",
        downloads: "380K",
        stars: "14.7K",
        date: "2023-06-08",
        tags: ["music-generation", "audio-synthesis", "controllable"],
        githubUrl: "https://github.com/facebookresearch/audiocraft",
        huggingtfaceUrl: "https://huggingface.co/facebook/musicgen-large"
    },
    {
        name: "LLaVA",
        author: "University of Wisconsin",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Large Language and Vision Assistant for visual instruction following.",
        plainDescription: "AI that can see and understand images while having conversations about them. Upload photos and ask questions, get descriptions, or request analysis.",
        useCases: ["Image Analysis", "Visual Q&A", "Photo Description", "Educational Assistance", "Accessibility Support"],
        inputModalities: ["Text", "Image"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 13B",
            memoryRequired: "14GB - 26GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "620K",
        stars: "15.2K",
        date: "2023-04-17",
        tags: ["vision-language", "instruction-following", "multimodal"],
        githubUrl: "https://github.com/haotian-liu/LLaVA",
        huggingtfaceUrl: "https://huggingface.co/liuhaotian/llava-v1.5-13b"
    },
    {
        name: "E5",
        author: "Microsoft",
        provider: "Hugging Face",
        category: "embedding",
        description: "Text embeddings by weakly-supervised contrastive pre-training.",
        plainDescription: "Advanced text understanding model that creates meaningful representations of text for search and similarity tasks across multiple languages.",
        useCases: ["Multilingual Search", "Document Retrieval", "Text Classification", "Semantic Matching", "Cross-lingual Tasks"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "110M - 335M",
            memoryRequired: "1GB - 3GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Sentence-Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "890K",
        stars: "1.8K",
        date: "2022-12-07",
        tags: ["text-embeddings", "retrieval", "contrastive-learning"],
        githubUrl: "https://github.com/microsoft/unilm/tree/master/e5",
        huggingtfaceUrl: "https://huggingface.co/intfloat/e5-large-v2"
    },
    {
        name: "Falcon",
        author: "Technology Innovation Institute",
        provider: "Hugging Face",
        category: "language",
        description: "Foundation language model trained on RefinedWeb dataset with strong performance.",
        plainDescription: "Powerful conversational AI that excels at creative writing, problem-solving, and multilingual tasks with efficient performance and open licensing.",
        useCases: ["Creative Writing", "Code Generation", "Multilingual Chat", "Content Creation", "Research Assistance"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 180B",
            memoryRequired: "14GB - 360GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "AWQ", "GPTQ"],
            toolCalling: "Yes",
            reasoning: "Strong",
            inputCost: "$0.50",
            outputCost: "$1.50"
        },
        license: "Apache-2.0",
        downloads: "1.1M",
        stars: "6.8K",
        date: "2023-05-24",
        tags: ["large-language-model", "refined-web", "multilingual"],
        githubUrl: "https://github.com/Technology-Innovation-Institute/falcon",
        huggingtfaceUrl: "https://huggingface.co/tiiuae/falcon-40b"
    },
    {
        name: "SAM",
        author: "Meta",
        provider: "Hugging Face",
        category: "vision",
        description: "Segment Anything Model - promptable segmentation system with zero-shot generalization.",
        plainDescription: "Precisely cuts out any object from images by clicking on it. Works on any image without training, perfect for photo editing and object isolation.",
        useCases: ["Photo Editing", "Medical Imaging", "Autonomous Driving", "Content Creation", "Quality Inspection"],
        inputModalities: ["Image"],
        outputModalities: ["Masks"],
        technicalSpecs: {
            parameters: "90M - 640M",
            memoryRequired: "4GB - 8GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "ONNX", "TensorRT"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "750K",
        stars: "42.1K",
        date: "2023-04-05",
        tags: ["segmentation", "zero-shot", "computer-vision"],
        githubUrl: "https://github.com/facebookresearch/segment-anything",
        huggingtfaceUrl: "https://huggingface.co/facebook/sam-vit-large"
    }
];

let filteredModels = [...modelsDatabase];

function renderModels(models) {
    const modelsGrid = document.getElementById('modelsGrid');
    modelsGrid.innerHTML = '';

    models.forEach(model => {
        const modelCard = createModelCard(model);
        modelsGrid.appendChild(modelCard);
    });

    updateStats(models.length);
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

function updateStats(filteredCount) {
    document.getElementById('totalModels').textContent = modelsDatabase.length;
    document.getElementById('filteredModels').textContent = filteredCount;
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

document.getElementById('searchInput').addEventListener('input', filterModels);
document.getElementById('categoryFilter').addEventListener('change', filterModels);
document.getElementById('licenseFilter').addEventListener('change', filterModels);
document.getElementById('providerFilter').addEventListener('change', filterModels);
document.getElementById('sortBy').addEventListener('change', sortModels);

renderModels(modelsDatabase);
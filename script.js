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
    },
    {
        name: "GPT-J",
        author: "EleutherAI",
        provider: "Hugging Face",
        category: "language",
        description: "6 billion parameter autoregressive language model trained on The Pile dataset.",
        plainDescription: "Open-source alternative to GPT-3 that can write stories, answer questions, generate code, and have conversations.",
        useCases: ["Text Generation", "Story Writing", "Code Completion", "Question Answering", "Creative Writing"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "6B",
            memoryRequired: "12GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "JAX"],
            toolCalling: "No",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.8M",
        stars: "9.2K",
        date: "2021-05-03",
        tags: ["autoregressive", "language-generation", "open-source"],
        githubUrl: "https://github.com/kingoflolz/mesh-transformer-jax",
        huggingtfaceUrl: "https://huggingface.co/EleutherAI/gpt-j-6b"
    },
    {
        name: "GPT-NeoX",
        author: "EleutherAI",
        provider: "Hugging Face",
        category: "language",
        description: "20 billion parameter language model implementing GPT-3 architecture.",
        plainDescription: "Large-scale AI for advanced text generation, creative writing, and complex reasoning tasks with high-quality outputs.",
        useCases: ["Research", "Content Generation", "Complex Reasoning", "Creative Writing", "Code Generation"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "20B",
            memoryRequired: "40GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "890K",
        stars: "7.1K",
        date: "2022-02-09",
        tags: ["large-language-model", "gpt-architecture", "research"],
        githubUrl: "https://github.com/EleutherAI/gpt-neox",
        huggingtfaceUrl: "https://huggingface.co/EleutherAI/gpt-neox-20b"
    },
    {
        name: "Alpaca",
        author: "Stanford",
        provider: "Hugging Face",
        category: "language",
        description: "Instruction-following model fine-tuned from LLaMA with 52K instruction-following examples.",
        plainDescription: "Fine-tuned AI assistant that follows instructions well, perfect for educational tasks and helpful conversations.",
        useCases: ["Education", "Instruction Following", "Tutoring", "Research Assistant", "Task Completion"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 65B",
            memoryRequired: "14GB - 130GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.3M",
        stars: "26.4K",
        date: "2023-03-13",
        tags: ["instruction-tuning", "fine-tuned", "educational"],
        githubUrl: "https://github.com/tatsu-lab/stanford_alpaca",
        huggingtfaceUrl: "https://huggingface.co/tatsu-lab/alpaca-7b-wdiff"
    },
    {
        name: "Vicuna",
        author: "UC Berkeley",
        provider: "Hugging Face",
        category: "language",
        description: "Open-source chatbot trained by fine-tuning LLaMA on user-shared conversations.",
        plainDescription: "Conversational AI trained on real chat data that excels at dialogue and maintaining context in long conversations.",
        useCases: ["Chatbots", "Customer Service", "Personal Assistant", "Dialogue Systems", "Interactive AI"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 33B",
            memoryRequired: "14GB - 66GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "FastChat"],
            toolCalling: "Yes",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "2.1M",
        stars: "33.8K",
        date: "2023-03-30",
        tags: ["conversational-ai", "chat", "fine-tuned"],
        githubUrl: "https://github.com/lm-sys/FastChat",
        huggingtfaceUrl: "https://huggingface.co/lmsys/vicuna-13b-v1.5"
    },
    {
        name: "OpenAssistant",
        author: "LAION",
        provider: "Hugging Face",
        category: "language",
        description: "Open-source conversational AI assistant trained to be helpful, harmless, and honest.",
        plainDescription: "Community-built AI assistant designed to be helpful and safe, trained with human feedback on diverse tasks.",
        useCases: ["General Assistant", "Educational Support", "Creative Tasks", "Code Help", "Information Retrieval"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "12B - 30B",
            memoryRequired: "24GB - 60GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "780K",
        stars: "35.9K",
        date: "2023-04-15",
        tags: ["assistant", "human-feedback", "community"],
        githubUrl: "https://github.com/LAION-AI/Open-Assistant",
        huggingtfaceUrl: "https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
    },
    {
        name: "MPT",
        author: "MosaicML",
        provider: "Hugging Face",
        category: "language",
        description: "MosaicML Pretrained Transformer with commercial use license and optimized training.",
        plainDescription: "Commercial-friendly AI model optimized for business use with strong performance on various language tasks.",
        useCases: ["Commercial Applications", "Business Intelligence", "Content Creation", "Data Analysis", "Enterprise AI"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 30B",
            memoryRequired: "14GB - 60GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "950K",
        stars: "5.1K",
        date: "2023-05-05",
        tags: ["commercial-use", "optimized", "enterprise"],
        githubUrl: "https://github.com/mosaicml/llm-foundry",
        huggingtfaceUrl: "https://huggingface.co/mosaicml/mpt-7b"
    },
    {
        name: "RedPajama",
        author: "Together",
        provider: "Hugging Face",
        category: "language",
        description: "Open reproduction of LLaMA trained on RedPajama dataset for full transparency.",
        plainDescription: "Transparent AI model with openly available training data, perfect for research and applications requiring full reproducibility.",
        useCases: ["Research", "Reproducible AI", "Academic Studies", "Transparent ML", "Educational Projects"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "3B - 7B",
            memoryRequired: "6GB - 14GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.2M",
        stars: "4.5K",
        date: "2023-04-17",
        tags: ["reproducible", "transparent", "open-data"],
        githubUrl: "https://github.com/togethercomputer/RedPajama-Data",
        huggingtfaceUrl: "https://huggingface.co/togethercomputer/RedPajama-INCITE-7B-Base"
    },
    {
        name: "Dolly",
        author: "Databricks",
        provider: "Hugging Face",
        category: "language",
        description: "Instruction-following large language model trained on human-generated instruction dataset.",
        plainDescription: "Business-focused AI assistant trained specifically for following complex instructions and enterprise applications.",
        useCases: ["Enterprise AI", "Instruction Following", "Business Analytics", "Data Processing", "Corporate Assistant"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "3B - 12B",
            memoryRequired: "6GB - 24GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "CC-BY-SA",
        downloads: "680K",
        stars: "6.3K",
        date: "2023-04-12",
        tags: ["instruction-following", "enterprise", "databricks"],
        githubUrl: "https://github.com/databrickslabs/dolly",
        huggingtfaceUrl: "https://huggingface.co/databricks/dolly-v2-12b"
    },
    {
        name: "FLAN-T5",
        author: "Google",
        provider: "Hugging Face",
        category: "language",
        description: "Instruction-tuned version of T5 trained on 1000+ tasks with detailed instructions.",
        plainDescription: "Google's instruction-following AI that excels at understanding and completing a wide variety of specific tasks and commands.",
        useCases: ["Task Automation", "Instruction Following", "Multi-task Learning", "Zero-shot Tasks", "Educational AI"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "80M - 11B",
            memoryRequired: "1GB - 22GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "TensorFlow", "JAX", "ONNX"],
            toolCalling: "Yes",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "2.3M",
        stars: "8.7K",
        date: "2022-10-20",
        tags: ["instruction-tuning", "multi-task", "google"],
        githubUrl: "https://github.com/google-research/t5x",
        huggingtfaceUrl: "https://huggingface.co/google/flan-t5-large"
    },
    {
        name: "ChatGLM",
        author: "Tsinghua University",
        provider: "Hugging Face",
        category: "language",
        description: "Bilingual conversational language model supporting both Chinese and English.",
        plainDescription: "Bilingual AI assistant optimized for Chinese and English conversations with strong reasoning and code generation abilities.",
        useCases: ["Multilingual Chat", "Chinese NLP", "Cross-lingual Tasks", "Educational Support", "Bilingual Content"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "6B - 130B",
            memoryRequired: "12GB - 260GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.5M",
        stars: "17.2K",
        date: "2023-03-14",
        tags: ["bilingual", "chinese", "conversational"],
        githubUrl: "https://github.com/THUDM/ChatGLM-6B",
        huggingtfaceUrl: "https://huggingface.co/THUDM/chatglm-6b"
    },
    {
        name: "WizardCoder",
        author: "WizardLM",
        provider: "Hugging Face",
        category: "language",
        description: "Code-specialized language model fine-tuned for programming tasks and code generation.",
        plainDescription: "Advanced coding AI that excels at programming tasks, debugging, code explanation, and software development across multiple languages.",
        useCases: ["Code Generation", "Programming Help", "Debugging", "Code Review", "Software Development"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "15B - 34B",
            memoryRequired: "30GB - 68GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "750K",
        stars: "9.8K",
        date: "2023-06-14",
        tags: ["code-generation", "programming", "wizard"],
        githubUrl: "https://github.com/nlpxucan/WizardLM",
        huggingtfaceUrl: "https://huggingface.co/WizardLM/WizardCoder-15B-V1.0"
    },
    {
        name: "Phind CodeLlama",
        author: "Phind",
        provider: "Hugging Face",
        category: "language",
        description: "Fine-tuned CodeLlama model optimized for code generation and programming assistance.",
        plainDescription: "Specialized programming AI optimized for real-world coding tasks with excellent performance on code completion and generation.",
        useCases: ["Code Completion", "Programming Assistant", "Software Development", "Algorithm Design", "Code Optimization"],
        inputModalities: ["Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "34B",
            memoryRequired: "68GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "GGUF", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Custom",
        downloads: "520K",
        stars: "2.1K",
        date: "2023-08-24",
        tags: ["code-generation", "programming", "fine-tuned"],
        githubUrl: "https://github.com/phind-com/Phind-CodeLlama-34B-v2",
        huggingtfaceUrl: "https://huggingface.co/Phind/Phind-CodeLlama-34B-v2"
    },
    {
        name: "ResNet",
        author: "Microsoft",
        provider: "PyTorch Hub",
        category: "vision",
        description: "Deep residual networks for image recognition with skip connections.",
        plainDescription: "Fundamental computer vision model that classifies images into thousands of categories with high accuracy and efficiency.",
        useCases: ["Image Classification", "Feature Extraction", "Transfer Learning", "Computer Vision Research", "Backbone Networks"],
        inputModalities: ["Image"],
        outputModalities: ["Classifications"],
        technicalSpecs: {
            parameters: "11M - 60M",
            memoryRequired: "2GB VRAM",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "ONNX", "TensorFlow", "CoreML"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "BSD-3-Clause",
        downloads: "5.2M",
        stars: "28.4K",
        date: "2015-12-10",
        tags: ["image-classification", "computer-vision", "backbone"],
        githubUrl: "https://github.com/pytorch/vision",
        huggingtfaceUrl: "https://huggingface.co/microsoft/resnet-50"
    },
    {
        name: "EfficientNet",
        author: "Google",
        provider: "TensorFlow Hub",
        category: "vision",
        description: "Efficient convolutional neural networks for image classification with optimal accuracy-efficiency trade-off.",
        plainDescription: "Highly efficient image classifier that achieves state-of-the-art accuracy while using significantly fewer computational resources.",
        useCases: ["Mobile Applications", "Edge Computing", "Efficient Classification", "Resource-Constrained Deployment", "Real-time Processing"],
        inputModalities: ["Image"],
        outputModalities: ["Classifications"],
        technicalSpecs: {
            parameters: "5M - 66M",
            memoryRequired: "1GB - 4GB VRAM",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["TensorFlow", "PyTorch", "ONNX", "TensorRT"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "3.8M",
        stars: "12.1K",
        date: "2019-05-28",
        tags: ["efficient", "mobile", "classification"],
        githubUrl: "https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet",
        huggingtfaceUrl: "https://huggingface.co/google/efficientnet-b7"
    },
    {
        name: "Vision Transformer (ViT)",
        author: "Google",
        provider: "Hugging Face",
        category: "vision",
        description: "Pure transformer applied to image patches for image classification without convolutions.",
        plainDescription: "Revolutionary computer vision model that treats images like text, using attention mechanisms for superior image understanding.",
        useCases: ["Image Classification", "Vision Research", "Transfer Learning", "Medical Imaging", "Fine-grained Recognition"],
        inputModalities: ["Image"],
        outputModalities: ["Classifications"],
        technicalSpecs: {
            parameters: "86M - 632M",
            memoryRequired: "4GB - 8GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "JAX", "TensorFlow", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "2.1M",
        stars: "18.3K",
        date: "2020-10-22",
        tags: ["transformer", "attention", "vision"],
        githubUrl: "https://github.com/google-research/vision_transformer",
        huggingtfaceUrl: "https://huggingface.co/google/vit-large-patch16-224"
    },
    {
        name: "DINOv2",
        author: "Meta",
        provider: "Hugging Face",
        category: "vision",
        description: "Self-supervised Vision Transformer for learning robust visual features without labels.",
        plainDescription: "Advanced computer vision model that learns to understand images without needing labeled training data, excellent for feature extraction.",
        useCases: ["Feature Extraction", "Self-supervised Learning", "Image Representation", "Transfer Learning", "Computer Vision Research"],
        inputModalities: ["Image"],
        outputModalities: ["Features"],
        technicalSpecs: {
            parameters: "22M - 1.1B",
            memoryRequired: "2GB - 8GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "890K",
        stars: "7.2K",
        date: "2023-04-14",
        tags: ["self-supervised", "feature-extraction", "vision-transformer"],
        githubUrl: "https://github.com/facebookresearch/dinov2",
        huggingtfaceUrl: "https://huggingface.co/facebook/dinov2-large"
    },
    {
        name: "ConvNeXt",
        author: "Meta",
        provider: "Hugging Face",
        category: "vision",
        description: "Modernized ConvNet that competes with Vision Transformers using pure convolutions.",
        plainDescription: "Modern convolutional neural network that combines the best of traditional CNN design with transformer innovations for excellent image recognition.",
        useCases: ["Image Classification", "Object Detection", "Semantic Segmentation", "Feature Extraction", "Computer Vision Backbone"],
        inputModalities: ["Image"],
        outputModalities: ["Classifications", "Features"],
        technicalSpecs: {
            parameters: "28M - 197M",
            memoryRequired: "3GB - 6GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.1M",
        stars: "12.8K",
        date: "2022-01-10",
        tags: ["convnet", "modern", "backbone"],
        githubUrl: "https://github.com/facebookresearch/ConvNeXt",
        huggingtfaceUrl: "https://huggingface.co/facebook/convnext-large-224"
    },
    {
        name: "Swin Transformer",
        author: "Microsoft",
        provider: "Hugging Face",
        category: "vision",
        description: "Hierarchical Vision Transformer using shifted windows for efficient computation.",
        plainDescription: "Efficient transformer for computer vision that processes images hierarchically, excellent for various vision tasks with manageable computational cost.",
        useCases: ["Object Detection", "Semantic Segmentation", "Instance Segmentation", "Image Classification", "Vision Backbone"],
        inputModalities: ["Image"],
        outputModalities: ["Classifications", "Features"],
        technicalSpecs: {
            parameters: "28M - 197M",
            memoryRequired: "3GB - 6GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "950K",
        stars: "11.2K",
        date: "2021-03-25",
        tags: ["hierarchical", "efficient", "windows"],
        githubUrl: "https://github.com/microsoft/Swin-Transformer",
        huggingtfaceUrl: "https://huggingface.co/microsoft/swin-large-patch4-window7-224"
    },
    {
        name: "DETR",
        author: "Meta",
        provider: "Hugging Face",
        category: "vision",
        description: "End-to-end object detection with transformers, eliminating need for anchor generation.",
        plainDescription: "Revolutionary object detection model that finds and identifies objects in images using transformers, providing clean bounding boxes and labels.",
        useCases: ["Object Detection", "Instance Segmentation", "Scene Understanding", "Autonomous Driving", "Surveillance"],
        inputModalities: ["Image"],
        outputModalities: ["Bounding Boxes", "Labels"],
        technicalSpecs: {
            parameters: "41M",
            memoryRequired: "4GB VRAM",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "680K",
        stars: "11.8K",
        date: "2020-05-26",
        tags: ["object-detection", "transformer", "end-to-end"],
        githubUrl: "https://github.com/facebookresearch/detr",
        huggingtfaceUrl: "https://huggingface.co/facebook/detr-resnet-50"
    },
    {
        name: "Mask R-CNN",
        author: "Meta",
        provider: "Detectron2",
        category: "vision",
        description: "Framework for object instance segmentation extending Faster R-CNN with mask prediction.",
        plainDescription: "Advanced computer vision model that not only detects objects but also creates precise pixel-level masks showing exactly where each object is located.",
        useCases: ["Instance Segmentation", "Medical Imaging", "Autonomous Vehicles", "Robotics", "Image Analysis"],
        inputModalities: ["Image"],
        outputModalities: ["Masks", "Bounding Boxes", "Labels"],
        technicalSpecs: {
            parameters: "44M",
            memoryRequired: "4GB VRAM",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Detectron2", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.8M",
        stars: "25.1K",
        date: "2017-03-20",
        tags: ["instance-segmentation", "object-detection", "mask"],
        githubUrl: "https://github.com/facebookresearch/detectron2",
        huggingtfaceUrl: "https://huggingface.co/facebook/maskrcnn-resnet50-fpn"
    },
    {
        name: "Wav2Vec2",
        author: "Meta",
        provider: "Hugging Face",
        category: "audio",
        description: "Self-supervised learning framework for speech representation from raw audio.",
        plainDescription: "Advanced speech recognition model that learns to understand speech without transcripts, excellent for low-resource languages and speech analysis.",
        useCases: ["Speech Recognition", "Audio Analysis", "Low-resource Languages", "Speech-to-Text", "Audio Classification"],
        inputModalities: ["Audio"],
        outputModalities: ["Text", "Features"],
        technicalSpecs: {
            parameters: "95M - 317M",
            memoryRequired: "2GB - 4GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "1.2M",
        stars: "16.4K",
        date: "2020-06-20",
        tags: ["self-supervised", "speech-recognition", "wav2vec"],
        githubUrl: "https://github.com/facebookresearch/fairseq",
        huggingtfaceUrl: "https://huggingface.co/facebook/wav2vec2-large-960h"
    },
    {
        name: "SpeechT5",
        author: "Microsoft",
        provider: "Hugging Face",
        category: "audio",
        description: "Unified-modal encoder-decoder pre-trained model for spoken language processing.",
        plainDescription: "Versatile speech AI that can both understand and generate speech, perfect for text-to-speech and speech enhancement applications.",
        useCases: ["Text-to-Speech", "Speech Enhancement", "Voice Conversion", "Speech Synthesis", "Audio Processing"],
        inputModalities: ["Text", "Audio"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "144M",
            memoryRequired: "2GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "750K",
        stars: "4.2K",
        date: "2021-10-18",
        tags: ["text-to-speech", "speech-synthesis", "unified"],
        githubUrl: "https://github.com/microsoft/SpeechT5",
        huggingtfaceUrl: "https://huggingface.co/microsoft/speecht5_tts"
    },
    {
        name: "VALL-E X",
        author: "Microsoft",
        provider: "GitHub",
        category: "audio",
        description: "Cross-lingual neural codec language model for zero-shot text-to-speech synthesis.",
        plainDescription: "Advanced voice cloning AI that can mimic any voice in multiple languages with just a few seconds of sample audio.",
        useCases: ["Voice Cloning", "Multilingual TTS", "Audiobook Creation", "Voice Acting", "Language Learning"],
        inputModalities: ["Text", "Audio"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "600M",
            memoryRequired: "8GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Medium",
            formats: ["PyTorch"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "120K",
        stars: "6.8K",
        date: "2023-07-31",
        tags: ["voice-cloning", "multilingual", "zero-shot"],
        githubUrl: "https://github.com/Plachtaa/VALL-E-X",
        huggingtfaceUrl: "https://huggingface.co/Plachta/VALL-E-X"
    },
    {
        name: "Bark",
        author: "Suno AI",
        provider: "Hugging Face",
        category: "audio",
        description: "Transformer-based text-to-audio model that generates speech, music, and sound effects.",
        plainDescription: "Creative audio AI that generates realistic speech, music, and sound effects from text descriptions with emotional expression and multilingual support.",
        useCases: ["Creative Audio", "Podcast Creation", "Sound Effects", "Multilingual Speech", "Audio Content"],
        inputModalities: ["Text"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "1.7B",
            memoryRequired: "12GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Slow",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "890K",
        stars: "31.2K",
        date: "2023-04-07",
        tags: ["text-to-audio", "creative", "multilingual"],
        githubUrl: "https://github.com/suno-ai/bark",
        huggingtfaceUrl: "https://huggingface.co/suno/bark"
    },
    {
        name: "Tortoise TTS",
        author: "James Betker",
        provider: "GitHub",
        category: "audio",
        description: "Multi-voice text-to-speech system focused on quality over speed.",
        plainDescription: "High-quality text-to-speech system that produces exceptionally natural-sounding voices, though slower than real-time generation.",
        useCases: ["High-Quality TTS", "Audiobook Production", "Voice Acting", "Content Creation", "Accessibility"],
        inputModalities: ["Text"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "2.8B",
            memoryRequired: "8GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Slow",
            formats: ["PyTorch"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "180K",
        stars: "11.5K",
        date: "2022-05-12",
        tags: ["high-quality", "text-to-speech", "multi-voice"],
        githubUrl: "https://github.com/neonbjb/tortoise-tts",
        huggingtfaceUrl: "https://huggingface.co/Tortoise-TTS"
    },
    {
        name: "FastSpeech2",
        author: "Microsoft",
        provider: "GitHub",
        category: "audio",
        description: "Fast and high-quality neural text-to-speech synthesis with controllable prosody.",
        plainDescription: "Fast, controllable text-to-speech system that allows fine control over speaking speed, pitch, and rhythm for natural voice synthesis.",
        useCases: ["Real-time TTS", "Voice Assistants", "Controllable Speech", "Interactive Applications", "Accessibility Tools"],
        inputModalities: ["Text"],
        outputModalities: ["Audio"],
        technicalSpecs: {
            parameters: "28M",
            memoryRequired: "1GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "320K",
        stars: "1.8K",
        date: "2020-06-08",
        tags: ["fast", "controllable", "real-time"],
        githubUrl: "https://github.com/ming024/FastSpeech2",
        huggingtfaceUrl: "https://huggingface.co/espnet/fastspeech2"
    },
    {
        name: "BLIP",
        author: "Salesforce",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Bootstrapping Language-Image Pre-training for unified vision-language understanding and generation.",
        plainDescription: "Versatile AI that understands both images and text, capable of describing images, answering visual questions, and generating captions.",
        useCases: ["Image Captioning", "Visual Question Answering", "Image-Text Retrieval", "Content Moderation", "Accessibility"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "224M",
            memoryRequired: "4GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "No",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "BSD-3-Clause",
        downloads: "1.1M",
        stars: "4.2K",
        date: "2022-01-28",
        tags: ["vision-language", "captioning", "vqa"],
        githubUrl: "https://github.com/salesforce/BLIP",
        huggingtfaceUrl: "https://huggingface.co/Salesforce/blip-image-captioning-large"
    },
    {
        name: "BLIP-2",
        author: "Salesforce",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Bootstrapped vision-language pre-training with frozen image encoders and language models.",
        plainDescription: "Advanced multimodal AI that connects powerful vision and language models for superior image understanding and conversational abilities.",
        useCases: ["Visual Conversations", "Image Analysis", "Educational AI", "Content Creation", "Research Assistant"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "2.7B - 7.8B",
            memoryRequired: "16GB - 32GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Limited",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "BSD-3-Clause",
        downloads: "680K",
        stars: "8.1K",
        date: "2023-01-30",
        tags: ["vision-language", "frozen-models", "conversation"],
        githubUrl: "https://github.com/salesforce/LAVIS",
        huggingtfaceUrl: "https://huggingface.co/Salesforce/blip2-opt-2.7b"
    },
    {
        name: "Flamingo",
        author: "DeepMind",
        provider: "GitHub",
        category: "multimodal",
        description: "Few-shot learning vision-language model for multimodal understanding tasks.",
        plainDescription: "Advanced AI that learns to understand new visual concepts from just a few examples, excellent for rapid adaptation to new tasks.",
        useCases: ["Few-shot Learning", "Visual Reasoning", "Multimodal Research", "Adaptive AI", "Educational Technology"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "80B",
            memoryRequired: "160GB",
            hardwareRequirement: "Multiple GPUs",
            inferenceSpeed: "Slow",
            formats: ["JAX"],
            toolCalling: "No",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "45K",
        stars: "3.1K",
        date: "2022-04-29",
        tags: ["few-shot", "vision-language", "deepmind"],
        githubUrl: "https://github.com/deepmind/flamingo",
        huggingtfaceUrl: "https://huggingface.co/deepmind/flamingo"
    },
    {
        name: "InstructBLIP",
        author: "Salesforce",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Instruction-tuned vision-language model for following complex visual instructions.",
        plainDescription: "Multimodal AI assistant that follows detailed instructions about images, perfect for complex visual analysis and educational applications.",
        useCases: ["Visual Instruction Following", "Educational AI", "Image Analysis", "Content Moderation", "Accessibility Support"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "7B - 13B",
            memoryRequired: "14GB - 26GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "BSD-3-Clause",
        downloads: "420K",
        stars: "5.7K",
        date: "2023-05-11",
        tags: ["instruction-following", "vision-language", "multimodal"],
        githubUrl: "https://github.com/salesforce/LAVIS",
        huggingtfaceUrl: "https://huggingface.co/Salesforce/instructblip-vicuna-7b"
    },
    {
        name: "LayoutLM",
        author: "Microsoft",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Pre-trained model for document understanding combining text, layout, and image information.",
        plainDescription: "Specialized AI for understanding documents that considers both text content and visual layout, perfect for forms, invoices, and structured documents.",
        useCases: ["Document Processing", "Form Understanding", "Invoice Analysis", "Information Extraction", "OCR Enhancement"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text", "Annotations"],
        technicalSpecs: {
            parameters: "113M - 344M",
            memoryRequired: "2GB - 4GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "Good",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "950K",
        stars: "3.8K",
        date: "2019-12-31",
        tags: ["document-understanding", "layout", "ocr"],
        githubUrl: "https://github.com/microsoft/unilm",
        huggingtfaceUrl: "https://huggingface.co/microsoft/layoutlm-base-uncased"
    },
    {
        name: "ALIGN",
        author: "Google",
        provider: "GitHub",
        category: "multimodal",
        description: "Large-scale noisy image-text alignment without manual data curation.",
        plainDescription: "Vision-language model trained on massive web data that excels at understanding relationships between images and text descriptions.",
        useCases: ["Image-Text Matching", "Content Discovery", "Visual Search", "Cross-modal Retrieval", "Multimodal Research"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "1.8B",
            memoryRequired: "8GB",
            hardwareRequirement: "GPU recommended",
            inferenceSpeed: "Fast",
            formats: ["TensorFlow", "JAX"],
            toolCalling: "No",
            reasoning: "Basic",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "180K",
        stars: "2.1K",
        date: "2021-02-11",
        tags: ["alignment", "noisy-data", "large-scale"],
        githubUrl: "https://github.com/google-research/google-research/tree/master/align",
        huggingtfaceUrl: "https://huggingface.co/google/align"
    },
    {
        name: "CogVLM",
        author: "Tsinghua University",
        provider: "Hugging Face",
        category: "multimodal",
        description: "Powerful open-source visual language model supporting image understanding and generation.",
        plainDescription: "Advanced Chinese-English bilingual AI that understands images and can have detailed conversations about visual content with strong reasoning abilities.",
        useCases: ["Bilingual Vision Chat", "Image Analysis", "Visual Reasoning", "Educational Support", "Multimodal Research"],
        inputModalities: ["Image", "Text"],
        outputModalities: ["Text"],
        technicalSpecs: {
            parameters: "17B",
            memoryRequired: "34GB",
            hardwareRequirement: "GPU required",
            inferenceSpeed: "Medium",
            formats: ["PyTorch", "Transformers"],
            toolCalling: "Yes",
            reasoning: "Strong",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "280K",
        stars: "5.2K",
        date: "2023-10-25",
        tags: ["bilingual", "vision-language", "reasoning"],
        githubUrl: "https://github.com/THUDM/CogVLM",
        huggingtfaceUrl: "https://huggingface.co/THUDM/cogvlm-chat-hf"
    },
    {
        name: "BGE",
        author: "BAAI",
        provider: "Hugging Face",
        category: "embedding",
        description: "General embedding model for text retrieval and representation learning.",
        plainDescription: "High-performance text embedding model optimized for search and retrieval tasks with excellent multilingual capabilities.",
        useCases: ["Semantic Search", "Document Retrieval", "Text Similarity", "RAG Systems", "Multilingual Search"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "33M - 335M",
            memoryRequired: "1GB - 3GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Very Fast",
            formats: ["PyTorch", "Sentence-Transformers", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "2.1M",
        stars: "4.8K",
        date: "2023-09-11",
        tags: ["embeddings", "retrieval", "multilingual"],
        githubUrl: "https://github.com/FlagOpen/FlagEmbedding",
        huggingtfaceUrl: "https://huggingface.co/BAAI/bge-large-en-v1.5"
    },
    {
        name: "Instructor",
        author: "HKU NLP",
        provider: "Hugging Face",
        category: "embedding",
        description: "Instruction-following text embedding model for diverse embedding tasks.",
        plainDescription: "Flexible embedding model that follows natural language instructions to create specialized embeddings for different tasks and domains.",
        useCases: ["Instruction-based Embeddings", "Task-specific Search", "Domain Adaptation", "Flexible Retrieval", "Custom Applications"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "110M - 335M",
            memoryRequired: "1GB - 3GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Sentence-Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "1.5M",
        stars: "1.9K",
        date: "2022-12-20",
        tags: ["instruction-following", "flexible", "task-specific"],
        githubUrl: "https://github.com/xlang-ai/instructor-embedding",
        huggingtfaceUrl: "https://huggingface.co/hkunlp/instructor-large"
    },
    {
        name: "Universal Sentence Encoder",
        author: "Google",
        provider: "TensorFlow Hub",
        category: "embedding",
        description: "Pre-trained encoder for generating sentence embeddings across multiple tasks.",
        plainDescription: "Google's versatile text encoder that creates meaningful sentence representations for various NLP tasks and multilingual applications.",
        useCases: ["Sentence Similarity", "Text Classification", "Clustering", "Semantic Search", "Cross-lingual Tasks"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "147M",
            memoryRequired: "1GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["TensorFlow", "TensorFlow Hub", "ONNX"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "Apache-2.0",
        downloads: "3.5M",
        stars: "8.2K",
        date: "2018-03-29",
        tags: ["universal", "sentence", "multilingual"],
        githubUrl: "https://github.com/tensorflow/hub",
        huggingtfaceUrl: "https://huggingface.co/google/universal-sentence-encoder"
    },
    {
        name: "SimCSE",
        author: "Princeton NLP",
        provider: "Hugging Face",
        category: "embedding",
        description: "Simple contrastive learning framework for sentence embeddings.",
        plainDescription: "Efficient method for creating high-quality sentence embeddings using contrastive learning, excellent for similarity tasks and clustering.",
        useCases: ["Sentence Similarity", "Semantic Textual Similarity", "Information Retrieval", "Clustering", "Duplicate Detection"],
        inputModalities: ["Text"],
        outputModalities: ["Embeddings"],
        technicalSpecs: {
            parameters: "110M - 335M",
            memoryRequired: "1GB - 3GB",
            hardwareRequirement: "CPU/GPU",
            inferenceSpeed: "Fast",
            formats: ["PyTorch", "Transformers", "Sentence-Transformers"],
            toolCalling: "No",
            reasoning: "None",
            inputCost: "Free",
            outputCost: "Free"
        },
        license: "MIT",
        downloads: "890K",
        stars: "2.8K",
        date: "2021-04-18",
        tags: ["contrastive", "similarity", "simple"],
        githubUrl: "https://github.com/princeton-nlp/SimCSE",
        huggingtfaceUrl: "https://huggingface.co/princeton-nlp/sup-simcse-bert-large-uncased"
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
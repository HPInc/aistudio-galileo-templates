# Text Generation with Galileo

## Content
- Overview
- Project Struture
- Setup
- Usage

## Overview 
This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It also integrates Galileo Prompt Quality for evaluation and logging, and supports multi-source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.

## Proect Struture
```
├── requirements.txt                     # Dependency file for installing required packages
├── README.md                            # Project overview and usage instructions
│
├── notebooks/
│   └── text-generation-with-langchain.ipynb   # Main notebook for generating scientific scripts using RAG
│
├── core/
│   ├── extract_text/
│   │   └── arxiv_search.py              # Module to search and download papers from arXiv
│   ├── analyzer/
│   │   └── scientific_paper_analyzer.py # Analyzer that uses LLM and retriever to process the paper
│   └── generator/
│       └── script_generator.py          # Class to manage section-by-section script generation
│

```

## Setup

### Quickstart

### Step 1: Create an AIstudio Project
1. Create a **New Project** in AI Studio
2. Select the template Text Generation with Langchain
3. Add a title description and relevant tags.

### Step 2: Verify Project Files
1. Launch a workspace.
2. Navigate to `03-text-generation/notebooks/text-generation-with-langchain.ipynb` to ensure all files were cloned correctly.


## Alternative Manual Setup

### Step 1: Create an AIStudio Project
1. Create a **New Project** in AI Studio.   
2. (Optional) Add a description and relevant tags.

### Step 2: Create a Workspace
1. Choose **Local GenAI** as the base image when creating the workspace.

### Step 3: Log Model
1. In the Datasets tab, click Add Dataset.
2. Upload the model file: `ggml-model-f16-Q5_K_M.gguf.`
3. The model will be available under the /datafabric directory in your workspace.

### Step 4: Verify Project Files  
1. In the Project Setup tab, under Setup, clone the project repository:
   ```
   git clone https://github.com/HPInc/aistudio-galileo-templates.git
   ```  
2. Navigate to `03-text-generation/notebooks/text-generation-with-langchain.ipynb` to ensure all files are cloned correctly after workspace creation.  

### Step 5: Use a Custom Kernel for Notebooks  
1. In Jupyter notebooks, select the **aistudio kernel** to ensure compatibility.

## Usage 
1. Open and execute the notebook `text-generation-with-langchain.ipynb`
2. In the **Run and Approve section**, you can customize prompts, add presentation sections, and view results directly in the Galileo Console.
```python
generator.add_section(
    name="title",
    prompt="Generate a clear and concise title for the presentation that reflects the content. Add a subtitle if needed. Respond using natural language only."
)
```



## Contact and Support  
- If you encounter issues, report them via GitHub by opening a new issue.  
- Refer to the **[AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview)** for detailed guidance and troubleshooting.  
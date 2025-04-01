# Text Generation with Galileo

## Overview 
This notebook implements a full Retrieval-Augmented Generation (RAG) pipeline for automatically generating a scientific presentation script. It integrates paper retrieval from arXiv, text extraction and chunking, embedding generation with HuggingFace, vector storage with ChromaDB, and context-aware generation using LLMs. It also integrates Galileo Prompt Quality for evaluation and logging, and supports multi-source model loading including local Llama.cpp, HuggingFace-hosted, and HuggingFace-cloud models like Mistral or DeepSeek.


## Features

- **arXiv Integration**: Searches and downloads scientific papers directly from the arXiv API using a user-defined query.
- **PDF Document Processing**: Extracts raw text from downloaded PDFs using PyMuPDF and parses it into structured documents.
- **Text Chunking with LangChain**: Splits long text into overlapping chunks using RecursiveCharacterTextSplitter for better context handling.
- **HuggingFace Embeddings**: Converts text chunks into semantic embeddings using HuggingFaceEmbeddings.
- **ChromaDB Vector Store**: Stores embeddings locally and enables fast semantic retrieval based on similarity search.
- **Scientific Paper Analyzer**: Builds a LangChain retriever-powered chain to analyze paper content and answer questions based on context.
- **ScriptGenerator Pipeline**: Generates a full scientific presentation script in sections (e.g., title, introduction, methodology) with human-in-the-loop approval.
- **Galileo PromptQuality Integration**: Logs and evaluates prompt performance using Galileo for continuous improvement and analysis.
- **Multi-Source Model Loader**: Supports:
- **Local inference with Llama.cpp** (e.g., LLaMA 2 7B),
- **Hugging Face Local model loading** (e.g., DeepSeek),
- **Hugging Face Cloud model access** (e.g., Mistral).
- **Modular Prompt Templates**: Each script section uses a dedicated prompt to ensure clarity and quality in the final presentation output.

## Instalation
Ensure all required dependencies are installed before running the notebook. Use the provided requirements.txt file:
```sh
pip install -r ../requirements.txt
```

## Usage

### 1. Environment & API Configuration
- Installs dependencies via requirements.txt.
- Sets up HuggingFace model caching.
- Loads config.yaml and secrets.yaml for API keys and model preferences.
- (Optional) Configures proxy for enterprise network environments.

### 2. Scientific Paper Retrieval & Processing
- Searches arXiv for papers using a user-defined query (e.g., "large language models").
- Downloads the corresponding PDF file.
- Extracts raw text using PyMuPDF.

### 3. Text Chunking & Embedding Generation
- Wraps the extracted text into Document objects with metadata.
- Splits the text into ~1200-character overlapping chunks using RecursiveCharacterTextSplitter.
- Computes embeddings using HuggingFaceEmbeddings.

### 4. Vector Store Setup (ChromaDB)
- Stores embeddings in a local Chroma vector database.
- Creates a retriever for similarity-based search over document chunks.

### 5. LLM Initialization & Paper Analysis
- Loads the model based on model_source (configurable as local, HF-local, or HF-cloud).
- Creates a ScientificPaperAnalyzer to answer questions using the retrieved context.
- Example:

```python
response = analyzer.analyze("What are the main findings of the paper?")
print(response)
```
### 6. Scientific Script Generation (Human-in-the-Loop)
- Uses ScriptGenerator to define and generate sections of a presentation:
- title, introduction, methodology, results, conclusion, references.
- Each section is crafted with a dedicated natural language prompt.
- All interactions are logged via Galileo PromptQuality for monitoring and evaluation.

### 7. Final Script Assembly
- Combines all sections into a cohesive presentation script:
```python
generator.run()
script = generator.get_final_script()
print("Final Script:\n", script)
```

## File Structure
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

## Key Functions
| Function Name                    | Description |
|----------------------------------|-------------|
| `ArxivSearcher.search_and_extract()`           | Searches arXiv using a keyword, downloads the paper, and extracts its raw text. |
| `RecursiveCharacterTextSplitter.split_documents()	` | Splits the extracted text into overlapping chunks for better contextual retrieval. |
| `HuggingFaceEmbeddings()` | Generates semantic embeddings from text chunks using Hugging Face models. |
| `Chroma.from_documents()` | Creates a local vector store and stores the text embeddings for similarity search. |
| `Chroma.as_retriever()` | Returns a retriever object that performs semantic similarity queries. |
| `ScientificPaperAnalyzer.analyze(query)` | Answers a scientific question using retrieved context from the paper. |
| `ScriptGenerator.add_section(name, prompt)` | Adds a custom prompt section to the final presentation script. |
| `ScriptGenerator.run()` | Runs the full pipeline to generate each script section sequentially. |
| `ScriptGenerator.get_final_script()` | Assembles and returns the complete scientific presentation script. |

## Conclusion
This scientific script generation notebook showcases a powerful and modular implementation of the Retrieval-Augmented Generation (RAG) architecture. By combining arXiv integration, HuggingFace embeddings, ChromaDB vector search, and LLM-powered generation, it enables the creation of rich, context-aware scientific presentations.

The integration with Galileo PromptQuality enhances transparency and evaluation, while the flexibility in model loading (local and cloud) makes the pipeline adaptable for various compute environments. Altogether, this solution serves as a strong foundation for building intelligent academic assistants, research automation tools, and educational content generators.
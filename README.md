# Notebook templates for integrating AI Studio and Galileo

In this repository, we provide a series of use cases to illustrate the integration between AI Studio and Galileo, often through the use of LangChain to orchestrate the language pipelines and allow the logging of metrics into Galileo.

## Repository Structure

The repository is organized into the following structure:

```
templates/
  ├── common/                   # Shared resources across notebooks
  │    ├── config.yaml          # Configuration settings
  │    ├── secrets.yaml         # API keys (gitignored, you must create this file and fill with the necessary API keys)
  │    └── data/                # Shared data files
  │
  ├── chatbot/                  # Chatbot template
  │    ├── chatbot-with-langchain.ipynb
  │    ├── requirements.txt
  │    ├── demo/                # UI demo output
  │    └── react_code/          # UI source code
  │
  ├── code-generation/          # Code generation template
  │    ├── code-generation-with-langchain.ipynb
  │    └── requirements.txt
  │
  ├── summarization/            # Summarization template
  │    ├── summarization-with-langchain.ipynb
  │    └── requirements.txt
  │
  └── text-generation/          # Text generation template
       ├── text-generation-with-langchain.ipynb
       └── requirements.txt
```

## Available Templates

### Chatbot (templates/chatbot/chatbot-with-langchain.ipynb)

In this simpler example, we implement a basic chatbot assistant to AI Studio, by means of a basic RAG pipeline. In this pipeline, we load information from a document (AIStudio documentation) into a vector database, then use this information to answer questions about AI Studio through proper prompting and use of LLMs. In the example, we illustrate different ways to load the model (locally and from cloud), and also illustrate how to use Galileo's callbacks to log information from the LangChain modules.

### Summarization (templates/summarization/summarization-with-langchain.ipynb)

For this use case, we extend the basic scenario to include more complex pre-processing of the input. In our scenario, we break an original transcript (which might be too long) into smaller topics (chunks with semantic relevance). A chain is then built to summarize the chunks --in parallel--, then joining them into a single summary in the end.

Also in this example, we illustrate how to work with:
* Personalized runs from Galileo (using EvaluateRuns)
* Personalized Metrics that runs locally (using CustomScorers)

### Code Generation (templates/code-generation/code-generation-with-langchain.ipynb)

This use case illustrates an example where the user accesses a git repository to serve as code reference. Some level of code understanding is used to index the available code from the repository. Based on this reference, the code generator uses in-cell prompts from the user in order to generate code in new notebook cells.

### Text Generation (templates/text-generation/text-generation-with-langchain.ipynb)

This use case shows a process to search for a scientific paper in ArXiv, then generating a presentation based on the content of this paper.

## Important notes when running these examples

To run the examples, you'll need to:

1. Install the requirements for each template using the provided requirements.txt file.
2. Set up your Galileo API key in the secrets.yaml file located in the templates/common directory.
3. Configure the model source in the config.yaml file according to your preferences.

## Different model alternatives

The notebooks support multiple model sources which can be configured in the common/config.yaml file:

- **local**: by loading the llama2-7b model from the asset downloaded on the project
- **hugging-face-local**: by downloading a DeepSeek model from Hugging Face and running locally
- **hugging-face-cloud**: by accessing the Mistral model through Hugging Face cloud API (requires HuggingFace API key)

# Notebook templates for integrating AI Studio and Galileo

In this folder, we bring a series of use cases to illustrate the integration between AI Studio and Galileo, often through the use of LangChain to orchestrate the language pipelines and allow the logging of metrics into Galileo. The following examples are available.

## Chatbot (chatbot-with-langchain.ipynb)

In this simpler example, we implement a basic chatbot assistant to AI Studio, by means of a basic RAG pipeline. In this pipeline, we load information from a document (AIStudio documentation) into a vector database, then use this information to answer questions about AI Studio through proper prompting and use of LLMs. In the example, we illustrate different ways to load the model (locally and from cloud), and also illustrate how to use Galileo's callbacks to log information from the LangChain modules

## Summarization (summarization-with-langchain.ipynb)

For this use case, we extend the basic scenario to include more complex pre-processing of the input. In our scenario, we break an original transcript (which might be too long) into smaller topics (chunks with semantic relevance). A chain is then built to summarize the chunks in parallel, then joining them into a single summary in the end.

Also in this example, we illustrate how to work with
* Personalized runs from Galileo (using EvaluateRuns)
* Personalized Metrics that runs locally (using CustomScorers)

## Code Generation (code-generation-with-langchain.ipynb)

This use case illustrates an example where the user accesses a git repository to serve as code reference. Some level of code understanding is used to index the available code from the repository. Based on this reference, the code generator uses in-cell prompts from the user in order to generate code in new notebook cells. 

## Text Generation (code-generation-with-langchain.ipynb)

This use case shows a process to search for a scientific paper in ArXiv, then generating a presentation based on the content of this paper.

# Different model alternatives

In the given examples, we provide different ways to access your Large Language models - all of them supported by our pre-configured workspace. Among them, we include:
* Access to OpenAI cloud API - Requires an API key from OpenAI
* Access to Hugging Face cloud API - Requires an API key from Hugging Face
* Access of Hugging Face models loaded locally, through transformer lib
* Access to models downloaded locally, through LlamaCPP library -  requires the project to have an asset call Llama7b, associated with the cloud S3 URI s3://dsp-demo-bucket/LLMs (public bucket)

# API Keys
At the moment, the user needs to create a file called secrets.yaml, with entries for desired keys. Integration with Galileo depends on creating users API key in the user interface and saving locally in this file, with the key Galileo.
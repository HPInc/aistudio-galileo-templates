# Experiments with AI Studio and Galileo

In the current version of the repository, we have two different notebooks that run full RAG pipelines and connect to Galileo to allow logging different metrics:
  * rag_cloud: Notebook that connects to OpenAI and uses GPT 3.5 for inference
  * rag_local: Notebook that uses LlamaCPP for inference on the RAG pipeline

Both of them are logging the results on the AIStudio_RAG project on Galileo, available here: https://console.hp.galileocloud.io/prompt/chains/db593b3a-3723-44ed-b53b-f03f0966560d

Each of the notebooks is composed by the following steps:

## Setup of the environment

In this step we install all the libraries that are necessary to run the RAG and log information into Galileo. They include
  * LangChain, langchain-community and related: Tool that supports running and managing the different modules for getting queries, retrieving relevant context and perform inference
  * ChromaDB - Vector database for retrieving chunks of documents more relevant to the search
  * Promptquality - library to connect to Galileo and log the metrics for the experiments

In the cloud example, we have also added the openai libraries, to allow accessing the cloud service that provides inference with GPT3. For the local model, we install llama-cpp, that allows for the local model to be loaded and inference to be made.

## Loading documents

A RAG pipeline has a knowledge base in the form of text. This knowledge base will be used to retrieve the more relevant parts that can help the model answering the given query. In this version, we are loading this knowledge base from PDF files, although it could be loaded from many other forms, such as web pages, pure text files, Word documents...

## Partitioning the knowledge base

As LLM have limited context size, only relevand parts of the documents should be input for the inference. For this reason, we split our knowledge base in chunks, using tools provided by langchain

## Embedding and creating the retrieval module

The next step involves using an embedding model to encode the chunks of text as numeric vectors. These embeddings are capable to capture semantic aspects of the text, and can be used to detect the similarity between different texts. All the embedded chunks are then added to the Vector Database (ChromaDB) that will be responsible to retrieve the ones more related to user inputs in the RAG pipeline

## The actual LLM 

For defining the large language model to be used, each notebook uses a different approach:
  * rag_cloud: In this case, the OpenAI API is used to perform the inference. This means that the RAG will connect to the cloud, send the prompt to OpenAI service, and get back the response given by ChatGPT. In this example, we are temporarily using a personal key for accessing OpenAI services (this key will be discontinued soon, so the user is welcome to change the key for a personal one)
  * rag_local: In this notebook, a model previously downloaded is loaded into the local GPU, to be run using LlamaCPP. This step is very resource-intensive - it loads the full model into GPU, and might take around 15 minutes to finish running.

## Defining the chain

In this stage, all the elements are put together, through langchaing, to allow the whole flow: User asks a question, ChromaDB retrieves a group of relevant chunks of information, the prompt template joins this information into an understandable prompt, the prompt is input to the model (either locally or as an API call to the cloud) and the response is returned as output

## Connecting to Galileo

For this last part, a callback is created to enable Galileo to retrieve the relevant metrics and log into the system. This will allow the visualization of the different runs oof the experiment inside Galileo project. Again, a personal, temporary key is being used to log into the Galileo - this key will be soon discontinued.

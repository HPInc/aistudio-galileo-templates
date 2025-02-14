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

# Important notes when running these examples
## API keys
At the moment, the user needs to create a file called secrets.yaml, with entries for desired keys. Integration with Galileo depends on creating users API key in the user interface and saving locally in this file, with the key Galileo.

## Resource issues
Several users have found issues when running this experiment due to the models exceeding Memory/GPU usage. To run these examples, we strongly recommend an NVidia GPU with 16+ GB of VRAM. We also recommend at least 16GB of RAM. Moreover, even with good hardware, we also recommend never running more than one model simultaneously: For instance, after starting a model in a notebook, restart the kernel or stop the workspace before running another notebook or model deployment.

## Network issues
Whenever corporate networks restrict SSH connections, one may need to set up HTTPS proxy to connect to external APIs - in our examples, logging data in Galileo web service requires SSH. If this is the case, one can setup a `proxy` variable in the config.yaml file before running the experiment - this will allow all the API calls to use the correct proxy.

## Llama issues
Some users have experienced issues related to the path not being found for the Llama7b model when running the experiments. If you experience this, first make sure that the model exists inside the workspace in the following location: `/home/jovyan/datafabric/Llama7b/ggml-model-f16-Q5_K_M.gguf`
If the file is not present at that location, please download it from the assets screen inside of your project. The assets page can be accessed by going into your project and clicking on the 'Assets' tab. Once here, click on the download button next to the Llama7b model in the assets list, which should take a few minutes to complete. Afterwards, restart the workspace and check the path above to see if the model was downloaded successfully.

# Different model alternatives

In the given examples, we provide different options as examples for  Large Language models - config.yaml file has a field *model_source*, which the user can choose between:
    * **local**: Uses local llama2-7b model, loaded from S3 as an asset in AI Studio Project
    * **hugging-face-local**: Downloads a deep-seek with 1.5B parameters and performs the inference locally
    * **hugging-face-cloud**: Uses Hugging Face API to access a Mistral 7b model. Requires a Hugging Face key registered in secrets.yaml (key HuggingFace)

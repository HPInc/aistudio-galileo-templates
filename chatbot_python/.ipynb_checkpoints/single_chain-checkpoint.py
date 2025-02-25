import yaml
import os

os.environ["HF_HOME"] = "/home/jovyan/local/hugging_face"
os.environ["HF_HUB_CACHE"] = "/home/jovyan/local/hugging_face/hub"

from trt_llm_langchain import TensorRTLangchain
from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, HuggingFaceEndpoint
from langchain.vectorstores import Chroma
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.document import Document
from typing import List
import promptquality as pq





with open('config.yaml') as file:
    config = yaml.safe_load(file)
    if "proxy" in config:
        os.environ["HTTPS_PROXY"] = config["proxy"]

file_path = (
    "data/AIStudioDoc.pdf"
)
pdf_loader = PyPDFLoader(file_path)
pdf_data = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
splits = text_splitter.split_documents(pdf_data)

embedding = HuggingFaceEmbeddings()

vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vectordb.as_retriever()

model_source = "tensorrt"
with open('config.yaml') as file:
    config = yaml.safe_load(file)
    if "model_source" in config:
        model_source = config["model_source"]

if model_source == "hugging-face-cloud":
    with open('secrets.yaml') as file:
        secrets = yaml.safe_load(file)
        huggingfacehub_api_token = secrets["HuggingFace"]
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=huggingfacehub_api_token,
        repo_id=repo_id,
    )
elif model_source == "hugging-face-local":
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
    llm = HuggingFacePipeline(pipeline=pipe)
elif model_source == "tensorrt":
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    llm = TensorRTLangchain(model_path = model_path)
else:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path="/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf",
        n_gpu_layers=30,
        n_batch=512,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,  
        callback_manager=callback_manager,
        verbose=False,
        stop=[],
        streaming=False,
        temperature=0.2,
    )

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([d.page_content for d in docs])

template = """You are an virtual Assistant for a Data Science platform called AI Studio. Answer the question based on the following context:

    {context}

    Question: {query}
    """
prompt = ChatPromptTemplate.from_template(template)

chain = {"context": retriever | format_docs, "query": RunnablePassthrough()} | prompt | llm | StrOutputParser()


import yaml

#########################################
# In order to connect to Galileo, create a secrets.yaml file in the same folder as this notebook
# This file should be an entry called Galileo, with the your personal Galileo API Key
# Galileo API keys can be created on https://console.hp.galileocloud.io/settings/api-keys
#########################################

with open('secrets.yaml') as file:
    secrets = yaml.safe_load(file)
    os.environ['GALILEO_API_KEY'] = secrets["Galileo"]

os.environ['GALILEO_CONSOLE_URL'] = "https://console.hp.galileocloud.io/" 

pq.login(os.environ['GALILEO_CONSOLE_URL'])

# Create callback handler
prompt_handler = pq.GalileoPromptCallback(
    project_name="Chatbot_template_demo",
    scorers=[pq.Scorers.context_adherence_luna, pq.Scorers.correctness, pq.Scorers.toxicity, pq.Scorers.sexist]
)

# Run your chain experiments across multiple inputs with the galileo callback
inputs = [
    "What is AI Studio",
    "How to create projects in AI Studio?"
    "How to monitor experiments?",
    "What are the different workspaces available?",
    "What, exactly, is a workspace?",
    "How to share my experiments with my team?",
    "Can I access my Git repository?",
    "Do I have access to files on my local computer?",
    "How do I access files on the cloud?",
    "Can I invite more people to my team?"
]
chain.batch(inputs, config=dict(callbacks=[prompt_handler]))

# publish the results of your run
prompt_handler.finish()
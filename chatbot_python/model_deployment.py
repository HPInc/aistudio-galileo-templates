import os
import yaml
import uuid
import base64
import mlflow
import pandas as pd
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma # langchain.vectorstores deprecated
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain.schema.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from mlflow.pyfunc import PythonModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec, ParamSchema, ParamSpec
import galileo_protect as gp
from galileo_protect import ProtectTool, ProtectParser, Ruleset
from galileo_observe import GalileoObserveCallback
import tensorrt_llm
from typing import Dict, Any
from langchain_core.language_models import LLM
from langchain_core.utils import pre_init


class TensorRTLangchain(LLM):
    client: Any = None  
    model_path: str
    sampling_params: Any = None
    
    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        try:
            import tensorrt_llm
        except ImportError:
            raise ImportError(
                "Could not import tensorrt-llm library. "
                "Please install the tensorrt-llm library or "
                "consider using workspaces based on the NeMo Framewok"
            )
        model_path = values["model_path"]
        values["client"] = tensorrt_llm.LLM(model=model_path)
        values["sampling_params"] = tensorrt_llm.SamplingParams(temperature=0.2, top_p=0.9, max_tokens=256)
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "tensorrt"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {"model_path": self.model_path}
    
    def _call(self, prompt, stop) -> str:
        output = self.client.generate(prompt, self.sampling_params)
        return output.outputs[0].text
        

def format_docs(docs: List[Document]) -> str:
    return "\n\n".join([doc.page_content for doc in docs if isinstance(doc.page_content, str)])

class AIStudioChatbotService(PythonModel):

    def load_config(self, context):
        secrets_path = context.artifacts["secrets"]
        config_path = context.artifacts["config"]
        self.docs_path = context.artifacts["docs"]
        print(f"Loading secrets.yaml file from the path: {secrets_path}")
        with open(secrets_path, "r") as file:
            secrets = yaml.safe_load(file)
        print(f"Loading config.yaml file from the path: {config_path}")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.model_config = {
            "galileo_key": secrets.get("Galileo", ""),
            "hf_key": secrets.get("HuggingFace", ""),
            "galileo_url": config.get("galileo_url", "https://console.hp.galileocloud.io/"),
            "proxy": config.get("proxy", None),
            "model_source": config.get("model_source", "local"),
            "observe_project": "Deployed_Chatbot_Observations",
            "protect_project": "Deployed_Chatbot_Protection",
            "local_model_path": "/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf"
        }
    
    def load_local_hf_model(self, context):
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, device=0)
        self.llm = HuggingFacePipeline(pipeline=pipe)        
        print("Using the local Deep Seek model downloaded from HuggingFace.")

    def load_cloud_hf_model(self, context):   
        self.llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=self.model_config["hf_key"],
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        )     
        print("Using the cloud Mistral model on HuggingFace.")
    
    def load_local_model(self, context):
        print("[INFO] Initializing local LlamaCpp model.")
        self.callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        self.llm = LlamaCpp(
            model_path=context.artifacts["models"],
            n_gpu_layers=30,
            n_batch=512,
            n_ctx=4096,
            max_tokens=1024,
            f16_kv=True,
            callback_manager=self.callback_manager,
            verbose=False,
            stop=[],
            streaming=False,
            temperature=0.2,
        )
        print("Using the local LlamaCpp model")

    def load_tensorrt(self, context):
        model_path = context.artifacts["models"]
        self.llm = TensorRTLangchain(model_path = model_path)        
    

    def load_model(self, context):
        if self.model_config["model_source"] == "local":
            self.load_local_model(context)
        elif self.model_config["model_source"] == "hugging-face-local":
            self.load_local_hf_model(context)
        elif self.model_config["model_source"] == "hugging-face-cloud":
            self.load_cloud_hf_model(context)
        elif self.model_config["model_source"] == "tensorrt":
            self.load_tensorrt(context)
        else:
            print("Incorrect source informed for the model")

    def load_vector_database(self):
        pdf_path = os.path.join(self.docs_path, "AIS_zDocs.pdf")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"The file 'AIStudioDoc.pdf' was not found at: {pdf_path}")
        print(f"Reading and processing the PDF file: {pdf_path}")

        pdf_loader = PyPDFLoader(pdf_path)
        pdf_data = pdf_loader.load()
        for doc in pdf_data:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        splits = text_splitter.split_documents(pdf_data)
        print(f"PDF split into {len(splits)} parts.")

    
        self.embedding = HuggingFaceEmbeddings()
        self.vectordb = Chroma.from_documents(documents=splits, embedding=self.embedding)
        self.retriever = self.vectordb.as_retriever()
        print("Vector database created successfully.")

    def load_prompt(self):
        self.prompt_str = """You are a virtual assistant for a Data Science platform called AI Studio. Answer the question based on the following context:
            {context}
            Question: {input}
            """
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)

    def load_chain(self):
        input_normalizer = RunnableLambda(lambda x: {"input": x} if isinstance(x, str) else x)
        retriever_runnable = RunnableLambda(lambda x: self.retriever.get_relevant_documents(x["input"]))
        format_docs_r = RunnableLambda(format_docs)
        extract_input = RunnableLambda(lambda x: x["input"])

        self.chain = (
            input_normalizer
            | RunnableMap({
                "context": retriever_runnable | format_docs_r,
                "input": extract_input
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def protect_chain(self):
        # Set up Galileo Protect
        project = gp.create_project(self.model_config["protect_project"])
        project_id = project.id
        print(f"Project created in Galileo Protect. Project ID: {project_id}")

        stage = gp.create_stage(name=f"{self.model_config['protect_project']}_stage1", project_id=project_id)
        stage_id = stage.id
        print(f"Stage created in Galileo Protect. Stage ID: {stage_id}")

        ruleset = Ruleset(
            rules=[
                gp.Rule(metric=gp.RuleMetrics.pii, operator=gp.RuleOperator.any, target_value=["ssn","phone_number"])
            ],
            action=gp.OverrideAction(choices=["Sorry, I cannot answer that question."])
        )
        protect_tool = ProtectTool(stage_id=stage_id, prioritized_rulesets=[ruleset], timeout=10)
        protect_parser = ProtectParser(chain=self.chain)
        self.protected_chain = protect_tool | protect_parser.parser

    def load_context(self, context):
        self.load_config(context)
        if self.model_config["proxy"] is not None:
            os.environ["HTTPS_PROXY"] = self.model_config["proxy"]
        os.environ["GALILEO_API_KEY"] = self.model_config["galileo_key"]
        os.environ["GALILEO_CONSOLE_URL"] = self.model_config["galileo_url"]

        self.load_model(context)
        self.load_vector_database()
        self.load_prompt()
        self.load_chain()
        self.protect_chain()
   
        self.monitor_handler = GalileoObserveCallback(project_name=self.model_config["observe_project"])
        print("Embeddings, vector database, LLM, Galileo Protect and Observer models successfully configured.")

        self.memory = []

    def add_pdf(self, base64_pdf):
        pdf_bytes = base64.b64decode(base64_pdf)
        temp_pdf_path = f"/tmp/{uuid.uuid4()}.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)

        pdf_loader = PyPDFLoader(temp_pdf_path)
        pdf_data = pdf_loader.load()
        for doc in pdf_data:
            if not isinstance(doc.page_content, str):
                doc.page_content = str(doc.page_content)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        new_splits = text_splitter.split_documents(pdf_data)

        embedding = HuggingFaceEmbeddings()
        vectordb = Chroma.from_documents(documents=new_splits, embedding=embedding)
        self.retriever = vectordb.as_retriever()

        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def get_prompt_template(self):
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def set_prompt_template(self, new_prompt):
        self.prompt_str = new_prompt
        self.prompt = ChatPromptTemplate.from_template(self.prompt_str)
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def reset_history(self):
        self.memory = []
        return {
            "chunks": [],
            "history": [],
            "prompt": self.prompt_str,
            "output": "",
            "success": True
        }

    def inference(self, context, user_query):
        response = self.protected_chain.invoke(
            {"input": user_query, "output": ""},
            config=dict(callbacks=[self.monitor_handler])
        )
        relevant_docs = self.retriever.get_relevant_documents(user_query)
        chunks = [doc.page_content for doc in relevant_docs]

        self.memory.append({"role": "User", "content": user_query})
        self.memory.append({"role": "Assistant", "content": response})

        return {
            "chunks": chunks,
            "history": [f"<{m['role']}> {m['content']}\n" for m in self.memory],
            "prompt": self.prompt_str,
            "output": response,
            "success": True
        }

    def predict(self, context, model_input, params):
        if params.get("add_pdf", False):
            result = self.add_pdf(model_input['document'][0])
        elif params.get("get_prompt", False):
            result = self.get_prompt_template()
        elif params.get("set_prompt", False):
            result = self.set_prompt_template(model_input['prompt'][0])
        elif params.get("reset_history", False):
            result = self.reset_history()
        else:
            result = self.inference(context, model_input['query'][0])

        return pd.DataFrame([result])

    @classmethod
    def log_model(cls, secrets_path, config_path, docs_path, model_folder=None, demo_folder="demo", utils_folder="utils"):
        if demo_folder and not os.path.exists(demo_folder):
            os.makedirs(demo_folder, exist_ok=True)

        input_schema = Schema([
            ColSpec("string", "query"),
            ColSpec("string", "prompt"),
            ColSpec("string", "document")
        ])
        output_schema = Schema([
            ColSpec("string", "chunks"),
            ColSpec("string", "history"),
            ColSpec("string", "prompt"),
            ColSpec("string", "output"),
            ColSpec("boolean", "success")
        ])
        param_schema = ParamSchema([
            ParamSpec("add_pdf", "boolean", False),
            ParamSpec("get_prompt", "boolean", False),
            ParamSpec("set_prompt", "boolean", False),
            ParamSpec("reset_history", "boolean", False)
        ])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=param_schema)

        artifacts = {"secrets": secrets_path, "config": config_path, "docs": docs_path, "demo": demo_folder, "utils": utils_folder}
        if model_folder:
            artifacts["models"] = model_folder

        mlflow.pyfunc.log_model(
            artifact_path="aistudio_chatbot_service",
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "PyPDF",
                "pyyaml",
                "tokenizers==0.20.3",
                "httpx==0.27.2",
                "langchain",
                "langchain-community",
                "langchain-huggingface",
                "chromadb",
                "promptquality",
                "galileo-protect",
                "galileo-observe"
            ]
        )
        print("Model and artifacts successfully registered in MLflow.")

print("Initializing experiment in MLflow.")
mlflow.set_experiment("AIStudioChatbot_Service")

secrets_path = "secrets.yaml"
config_path = "config.yaml"
docs_path = "data"
model_folder = "/home/jovyan/local/mistral-7b-engine"
demo_folder = "demo"   
utils_folder = "utils"

# Ensure the demo folder exists before logging model
if demo_folder and not os.path.exists(demo_folder):
    os.makedirs(demo_folder, exist_ok=True)

if not os.path.exists(secrets_path):
    raise FileNotFoundError(f"secrets.yaml file not found in path: {os.path.abspath(secrets_path)}")
if not os.path.exists(docs_path):
    raise FileNotFoundError(f"'data' folder not found in path: {os.path.abspath(docs_path)}")

with mlflow.start_run(run_name="AIStudioChatbot_Service_Run") as run:
    AIStudioChatbotService.log_model(
        secrets_path=secrets_path,
        config_path=config_path,
        docs_path=docs_path,
        demo_folder=demo_folder,
        utils_folder=utils_folder,
        model_folder=model_folder
    )
    model_uri = f"runs:/{run.info.run_id}/aistudio_chatbot_service"
    mlflow.register_model(
        model_uri=model_uri,
        name="Chatbot-tensorRT",
    )
    print(f"Registered model with execution ID: {run.info.run_id}")
    print(f"Model registered successfully. Run ID: {run.info.run_id}")
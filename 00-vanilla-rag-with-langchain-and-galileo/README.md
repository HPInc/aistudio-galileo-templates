# Chatbot with LangChain, RAG, and Galileo

## 📚 Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Contact & Support](#contact--support)

---

## 🧠 Overview

This project is an AI-powered chatbot built using **LangChain**, **RAG (Retrieval-Augmented Generation)**, and **Galileo** for model evaluation, protection, and observability. It leverages the **Z by HP AI Studio Local GenAI image** and the **LLaMA2-7B** model to generate contextual and document-grounded answers to user queries.

---

## 🗂 Project Structure

```
├── README.md
├── core
│   └── chatbot_service
│       ├── __init__.py
│       └── chatbot_service.py
├── data
│   └── AIStudioDoc.pdf
├── demo
│   ├── assets
│   ├── index.html
│   └── source
├── notebooks
│   └── chatbot-with-langchain.ipynb
├── configs
│   ├── config.yaml
│   └── secrets.yaml
└── requirements.txt
```

---

## ⚙️ Setup

### Step 1: Create an AI Studio Project

- Create a new project in [Z by HP AI Studio](https://zdocs.datascience.hp.com/docs/aistudio/overview).
- (Optional) Add a description and relevant tags.

### Step 2: Set Up a Workspace

- Choose **Local GenAI** as the base image.

### Step 3: Clone the Repository

```bash
https://github.com/HPInc/aistudio-galileo-templates.git
```

- Ensure all files are available after workspace creation.

### Step 4: Add the Model to Workspace

- Download the **LLaMA2-7B** model from AWS S3:
  - **S3 URI**: `s3://149536453923-hpaistudio-public-assets/llama2-7b`
  - **Region**: `us-west-2`
- Make sure that the model in the `datafabric` folder inside your workspace.

### Step 5: Configure Secrets and Paths

- Add your API keys to the `secrets.yaml` file under the `configs` folder:
  - `HUGGINGFACE_API_KEY`
  - `GALILEO_API_KEY`
- Edit `config.yaml` with relevant configuration details.

---

## 🚀 Usage

### Step 1: Run the Notebook

Execute the notebook inside the `notebooks` folder:

```bash
notebooks/chatbot-with-langchain.ipynb
```

This will:

- Run the full RAG pipeline
- Integrate Galileo evaluation, protection, and observability
- Register the model in MLflow

### Step 2: Deploy the Chatbot Service

- Go to **Deployments > New Service** in AI Studio.
- Name the service and select the registered model.
- Choose a model version and enable **GPU acceleration**.
- Start the deployment.
- Once deployed, access the **Swagger UI** via the Service URL.
- From the Swagger page, click the demo link to interact with the locally deployed chatbot via UI.

---

## 📞 Contact & Support

- 💬 For issues or questions, please [open a GitHub issue](https://github.com/HPInc/aistudio-galileo-templates/issues).
- 📘 Refer to the official [AI Studio Documentation](https://zdocs.datascience.hp.com/docs/aistudio/overview) for detailed instructions and troubleshooting tips.

---

> Built with ❤️ using LangChain, Galileo, and Z by HP AI Studio.


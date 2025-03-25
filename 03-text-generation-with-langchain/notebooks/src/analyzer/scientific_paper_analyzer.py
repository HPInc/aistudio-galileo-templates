import logging
from typing import List, Optional
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.schema import Document, StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, Runnable
from langchain.vectorstores.base import VectorStoreRetriever


class ScientificPaperAnalyzer:
    """
    Class for analyzing scientific papers using LLMs and vector retrievers.
    """

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        llm: BaseChatModel,
        prompt_template: Optional[ChatPromptTemplate] = None,
        logging_enabled: bool = False
    ):
        """
        Initializes the class with a retriever, an LLM, and an optional prompt template.

        Args:
            retriever (VectorStoreRetriever): Retriever with get_relevant_documents method.
            llm (BaseChatModel): Language model instance.
            prompt_template (ChatPromptTemplate, optional): Custom prompt template. Uses default if None.
            logging_enabled (bool): If True, enables logging output.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        else:
            logging.disable(logging.CRITICAL)

        self.chain = self._build_chain()

    def _default_prompt(self) -> ChatPromptTemplate:
        template = """You are tasked with analyzing a scientific paper and responding to a series of steps or questions based on the paper's content. 
Your goal is to provide accurate, contextual responses for each step, drawing from the paper's information and your own knowledge when necessary.
Here is the paper you will be analyzing:
    {context}

1. Read the step carefully.
2. Search for relevant information in the paper that addresses the step.
3. If the paper contains information directly related to the step, use that information to formulate your response.
4. If the paper does not contain information directly related to the step, but the topic is related to the paper's content, use your own knowledge to provide a response that is consistent with the paper's context and subject matter.
    Question: {question}
"""
        return ChatPromptTemplate.from_template(template)

    def _format_docs(self, docs: List[Document]) -> str:
        content = "\n\n".join([d.page_content for d in docs])
        if self.logging_enabled:
            self.logger.info(f"Formatted {len(docs)} documents into context.")
        return content

    def _query_retriever(self, query: str) -> List[Document]:
        docs = self.retriever.get_relevant_documents(query)
        if self.logging_enabled:
            self.logger.info(f"Retrieved {len(docs)} documents for query: '{query}'")
        return docs

    def _build_chain(self) -> Runnable:
        if self.logging_enabled:
            self.logger.info("Building the LangChain chain...")
        return {
            "context": lambda inputs: self._format_docs(self._query_retriever(inputs["query"])),
            "question": RunnablePassthrough()
        } | self.prompt_template | self.llm | StrOutputParser()

    def analyze(self, question: str) -> str:
        if self.logging_enabled:
            self.logger.info(f"Analyzing question: '{question}'")
        result = self.chain.invoke({"query": question, "question": question})
        if self.logging_enabled:
            self.logger.info("Chain execution completed.")
        return result

    def get_chain(self) -> Runnable:
        """
        Returns the internal chain for external composition.
        """
        return self.chain

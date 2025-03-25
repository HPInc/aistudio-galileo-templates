import logging
import re
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
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self._default_prompt()
        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled

        # Detect automatically if the LLM might be DeepSeek (heuristic)
        self.is_deepseek = "deepseek" in str(type(llm)).lower()

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
            if self.is_deepseek:
                self.logger.info("Auto-detected DeepSeek model.")
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

    

    def _clean_deepseek_output(self, text: str) -> str:
        """
        Removes repeated prompt or dictionary-like structures 
        specifically for DeepSeek outputs.
        """
        pattern = r"(?s)Human:.*?Question:\s*\{.*?\}\s*"
        cleaned = re.sub(pattern, "", text)

        steps_pattern = r"(?s)1\.\s*Read the step.*?subject matter\.\s*"
        cleaned = re.sub(steps_pattern, "", cleaned)

        return cleaned.strip()

    def _clean_generic_output(self, text: str) -> str:
        """
        For non-DeepSeek models:
        1) Remove 'Response:' prefix se existir
        2) Remove duplicações exatas de linhas consecutivas
        """
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line_no_response = re.sub(r"^Response:\s*", "", line.strip())
            cleaned_lines.append(line_no_response)

        deduped = []
        for line in cleaned_lines:
            if not deduped or line.strip() != deduped[-1].strip():
                deduped.append(line)
        final_text = "\n".join(deduped).strip()

        return final_text

    

    def analyze(self, question: str) -> str:
        """
        Executes the chain and returns only the final answer text,
        removing repeated context or duplication as needed.
        """
        if self.logging_enabled:
            self.logger.info(f"Analyzing question: '{question}'")

        result = self.chain.invoke({"query": question, "question": question})

        if isinstance(result, dict) and "text" in result:
            result = result["text"]
        else:
            result = str(result)

        result = result.strip()

        if self.is_deepseek:
            cleaned = self._clean_deepseek_output(result)
        else:
            cleaned = self._clean_generic_output(result)

        if self.logging_enabled:
            self.logger.info("Chain execution completed.")
            self.logger.info(f"Final cleaned response: {cleaned[:300]}...")

        return cleaned

    def get_chain(self) -> Runnable:
        """
        Returns the internal chain for external composition.
        """
        return self.chain

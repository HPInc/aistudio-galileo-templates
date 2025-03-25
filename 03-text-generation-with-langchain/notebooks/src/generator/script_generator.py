import logging
from typing import Callable, List, Optional
import promptquality as pq
from langchain.schema.runnable import Runnable


class ScriptGenerator:
    """
    Orchestrates the generation of an academic script with step-by-step approval.

    Attributes:
        chain (Runnable): The LangChain execution chain.
        scorers (List[Callable]): List of PromptQuality scoring functions.
        logging_enabled (bool): Whether to enable logging output.
    """

    def __init__(
        self,
        chain: Runnable,
        scorers: Optional[List[Callable]] = None,
        logging_enabled: bool = False
    ):
        """
        Initializes the ScriptGenerator.

        Args:
            chain (Runnable): The LangChain chain to use for generation.
            scorers (List[Callable], optional): PromptQuality scorers. Default includes 3 common metrics.
            logging_enabled (bool): Enable or disable logging output.
        """
        self.chain = chain
        self.sections = []
        self.results = {}
        self.scorers = scorers or [
            pq.Scorers.context_adherence_plus,
            pq.Scorers.correctness,
            pq.Scorers.prompt_perplexity
        ]

        self.logger = logging.getLogger(__name__)
        self.logging_enabled = logging_enabled

        if logging_enabled:
            logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        else:
            logging.disable(logging.CRITICAL)

    def add_section(self, name: str, template: str, question: str, k: int = 10):
        """
        Adds a new section to the script generation pipeline.

        Args:
            name (str): Section identifier (e.g., 'title').
            template (str): Prompt template used as input context.
            question (str): Instruction or specific question for the model.
            k (int): Number of documents to retrieve (placeholder, not used here yet).
        """
        self.sections.append({
            "name": name,
            "template": template,
            "question": question,
            "k": k
        })
        if self.logging_enabled:
            self.logger.info(f"Section '{name}' added.")

    def _run_and_approve(self, section) -> str:
        """
        Runs a section's prompt and waits for manual approval.

        Args:
            section (dict): Section with keys: name, template, question, k.

        Returns:
            str: Approved result text.
        """
        while True:
            prompt_handler = pq.GalileoPromptCallback(scorers=self.scorers)

            if self.logging_enabled:
                self.logger.info(f"Generating output for section: {section['name']}")

            result = self.chain.batch(
                [{"query": section["template"], "question": section["question"]}],
                config=dict(callbacks=[prompt_handler])
            )

            if not result:
                if self.logging_enabled:
                    self.logger.warning(f"No result generated for section: {section['name']}")
                continue

            print(f"\n>>> [{section['name']}] Result:\n{result[0]}\n")
            approval = input("Approve the result? (y/n): ").strip().lower()
            if approval == 'y':
                prompt_handler.finish()
                if self.logging_enabled:
                    self.logger.info(f"Result approved for section: {section['name']}")
                return result[0]

            print("Result not approved. Regenerating...\n")
            if self.logging_enabled:
                self.logger.info(f"Retrying generation for section: {section['name']}")

    def run(self):
        """
        Executes all configured sections, asking for approval in each step.
        """
        for section in self.sections:
            print(f"Running section: {section['name']}")
            if self.logging_enabled:
                self.logger.info(f"Starting section: {section['name']}")
            self.results[section["name"]] = self._run_and_approve(section)

    def get_final_script(self) -> str:
        """
        Returns the concatenated final script.

        Returns:
            str: Combined output from all approved sections.
        """
        if self.logging_enabled:
            self.logger.info("Final script generated.")
        return "\n\n".join(self.results.values())

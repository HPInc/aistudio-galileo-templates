import logging
from typing import Callable, List, Optional
import promptquality as pq
from langchain.schema.runnable import Runnable


class ScriptGenerator:
    """
    Orchestrates the generation of an academic script with step-by-step approval.
    """

    def __init__(
        self,
        chain: Runnable,
        scorers: Optional[List[Callable]] = None,
        logging_enabled: bool = False,
    ):
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

    def add_section(self, name: str, prompt: str):
        self.sections.append({
            "name": name,
            "prompt": prompt
        })
        if self.logging_enabled:
            self.logger.info(f"Section '{name}' added with prompt:\n{prompt}")

    def _run_and_approve(self, section) -> str:
        while True:
            prompt_handler = pq.GalileoPromptCallback(scorers=self.scorers)

            if self.logging_enabled:
                self.logger.info(f"Generating output for section: {section['name']}")
                self.logger.info(f"Prompt sent to model:\n{section['prompt']}")

            result = self.chain.batch(
                [{"prompt": section["prompt"]}],
                config=dict(callbacks=[prompt_handler])
            )

            if not result:
                if self.logging_enabled:
                    self.logger.warning(f"No result generated for section: {section['name']}")
                continue

            raw_text = result[0]

            if self.logging_enabled:
                self.logger.info(f"Raw model response for section '{section['name']}':\n{raw_text[:300]}...")

            print(f"\n>>> [{section['name']}] Result:\n{raw_text}\n")
            approval = input("Approve the result? (y/n): ").strip().lower()
            if approval == 'y':
                prompt_handler.finish()
                if self.logging_enabled:
                    self.logger.info(f"Result approved for section: {section['name']}")
                return raw_text

            print("Result not approved. Regenerating...\n")
            if self.logging_enabled:
                self.logger.info(f"Retrying generation for section: {section['name']}")

    def run(self):
        for section in self.sections:
            print(f"Running section: {section['name']}")
            if self.logging_enabled:
                self.logger.info(f"Starting section: {section['name']}")
            self.results[section["name"]] = self._run_and_approve(section)

    def get_final_script(self) -> str:
        if self.logging_enabled:
            self.logger.info("Final script generated.")
        return "\n\n".join(self.results.values())

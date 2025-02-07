import time
from typing import Any, Dict, List
import promptquality as pq
from promptquality import NodeRow, NodeType

class GalileoChainRunner:
    """
    A class that runs a chain with Galileo using built-in default templates,
    and provides a manual approval step for each section.

    The default templates and associated questions for each section are defined
    within the class, so you don't need to pass them manually.
    """

    def __init__(self, chain: Any, vectordb: Any, k: int = 10) -> None:
        """
        Initializes the GalileoChainRunner with a chain, vector database, and an optional 'k' parameter.
        
        :param chain: The chain instance that implements a 'batch' method.
        :param vectordb: The vector database instance.
        :param k: Number of documents to be retrieved (default is 10).
        """
        self.chain = chain
        self.vectordb = vectordb
        self.k = k

        self.templates: Dict[str, str] = {
            "title": (
                "Generate a title for the presentation that is clear, concise, and reflects the content. "
                "Add a subtitle if needed."
            ),
            "introduction": (
                "Generate an introduction that includes:\n"
                "- Contextualization of the general theme.\n"
                "- Relevance of the topic, both academically and practically.\n"
                "- A brief literature review.\n"
                "- A clear definition of the research problem.\n"
                "- The specific objectives of the research.\n"
                "- Hypotheses (if applicable)."
            ),
            "methodology": (
                "Generate the methodology section, including:\n"
                "- Research Design (e.g., experimental, descriptive, exploratory).\n"
                "- Sample and Population details.\n"
                "- Data Collection methods.\n"
                "- Instruments used for data collection.\n"
                "- Data Analysis techniques."
            ),
            "results": (
                "Generate the results section, including:\n"
                "- Presentation of Data with visual aids like graphs and tables.\n"
                "- Initial Interpretation of the data.\n"
                "- Comparison with Hypotheses (if applicable)."
            ),
            "conclusion": (
                "Generate the conclusion of the study, including:\n"
                "- Synthesis of Results.\n"
                "- Response to the Research Problem.\n"
                "- Study Contributions.\n"
                "- Final Reflection on the study's impact or practical recommendations."
            ),
            "references": (
                "Generate the list of references for the study, ensuring that:\n"
                "- All sources cited in the presentation are included.\n"
                "- The references are formatted according to a specific style (APA, MLA, Chicago)."
            )
        }

        self.questions: Dict[str, str] = {
            "title": "create a title",
            "introduction": "generate an introduction",
            "methodology": "generate the methodology",
            "results": "generate the results",
            "conclusion": "generate the conclusion",
            "references": "generate the references"
        }

    def run_and_approve(self, variable_name: str, template: str, question: str) -> str:
        """
        Executes the chain using the provided template and question, and then asks the user
        to approve the result. If the result is not approved, it re-runs the chain until an approved result is obtained.

        :param variable_name: Name of the variable (for printing purposes).
        :param template: Prompt template used as context to generate text.
        :param question: Specific task the model should perform.
        :return: The approved generated text.
        """
        while True:
            prompt_handler = pq.GalileoPromptCallback(
                scorers=[
                    pq.Scorers.context_adherence_plus,
                    pq.Scorers.correctness,
                    pq.Scorers.prompt_perplexity
                ]
            )

            result = self.chain.batch(
                [{"query": template, "question": question}],
                config=dict(callbacks=[prompt_handler])
            )

            if not result:
                continue

            print(f"{variable_name}: {result[0]}")

            approval = input("Approve the result? (y/n): ").strip().lower()
            if approval == 'y':
                prompt_handler.finish()
                return result[0]

            print("Result not approved, generating again...\n")

    def run_section(self, section: str) -> str:
        """
        Runs a specific section (using its default template and question) and returns the approved result.

        :param section: The section to run. Must be one of the keys in the default templates.
        :return: The approved generated text for that section.
        :raises ValueError: If the section is not recognized.
        """
        if section not in self.templates or section not in self.questions:
            raise ValueError(f"Section '{section}' not recognized. Available sections: {list(self.templates.keys())}")
        return self.run_and_approve(section, self.templates[section], self.questions[section])

    def run_all_sections(self) -> str:
        """
        Runs all default sections in sequence and returns a final script combining all results.

        :return: The final combined script.
        """
        title_result = self.run_section("title")
        introduction_result = self.run_section("introduction")
        methodology_result = self.run_section("methodology")
        results_result = self.run_section("results")
        conclusion_result = self.run_section("conclusion")
        references_result = self.run_section("references")

        final_script = (
            f"{title_result}\n\n"
            f"{introduction_result}\n\n"
            f"{methodology_result}\n\n"
            f"{results_result}\n\n"
            f"{conclusion_result}\n\n"
            f"{references_result}"
        )
        return final_script



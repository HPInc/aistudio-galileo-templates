# text_generation_service.py
# -*- coding: utf-8 -*-
"""
MLflow pyfunc that searches for articles on arXiv, summarizes them, and (optionally)
generates a script from the summaries.

Per-line flags:
    do_extract   – downloads PDFs and converts them to text
    do_analyze   – calls the ScientificPaperAnalyzer
    do_generate  – calls the ScriptGenerator

Optional extra fields:
    analysis_prompt   – free-form prompt to pass to .analyze()
    generation_prompt – initial prompt for the ScriptGenerator
"""
from __future__ import annotations

import json
import time
import logging
import sys
import mlflow
from pathlib import Path
import pandas as pd
from mlflow.types import Schema, ColSpec
from mlflow.models import ModelSignature

_loglevel_path = Path(__file__).parent / ".loglevel"
LOGLEVEL = (
    _loglevel_path.read_text().strip()
    if _loglevel_path.exists()
    else "INFO"
)
logging.basicConfig(
    level=logging.getLevelName(LOGLEVEL),
    format="%(asctime)s — %(levelname)s — %(message)s",
)


def add_project_dirs_to_syspath() -> tuple[Path, Path | None]:
    this_file = Path(__file__).resolve()
    core_path = this_file.parent.parent
    (core_path / "__init__.py").touch(exist_ok=True)
    sys.path.insert(0, str(core_path))

    src_path: Path | None = None
    for parent in [core_path, *core_path.parents]:
        cand = parent / "src"
        if cand.is_dir():
            src_path = cand
            sys.path.insert(0, str(src_path))
            break

    repo_root = core_path.parent.parent
    sys.path.insert(0, str(repo_root))
    return core_path, src_path




def load_context_and_init(artifacts: dict[str, str]):
    from src.utils import configure_hf_cache, configure_proxy, load_config_and_secrets
    from langchain.callbacks.manager import CallbackManager
    from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
    from langchain_community.llms import LlamaCpp

    cfg_dir = Path(artifacts["config"]).parent
    config_path = cfg_dir / "config.yaml"
    secrets_path = cfg_dir / "secrets.yaml"
    config, _ = load_config_and_secrets(config_path, secrets_path)

    model_path = artifacts.get("llm")
    if not model_path:
        raise RuntimeError("*.gguf artifact not found!")
    logging.info("🔹 LlamaCpp → %s", model_path)

    configure_hf_cache()
    configure_proxy(config)

    t0 = time.perf_counter()
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=1,             
        n_batch=256,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,
        temperature=0.2,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=False,
        streaming=False,
    )
    logging.info("🔹 LLM loaded in %.1fs", time.perf_counter() - t0)
    return llm



class TextGenerationService(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        add_project_dirs_to_syspath()
        self.llm = load_context_and_init(context.artifacts)

    def _analyze_papers(
        self,
        papers: list[dict],
        analysis_prompt: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        """Creates an ad-hoc retriever and calls the ScientificPaperAnalyzer."""
        if not papers:
            return ""

        from langchain.schema import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from core.analyzer.scientific_paper_analyzer import ScientificPaperAnalyzer

        docs = [
            Document(page_content=p["text"], metadata={"title": p["title"]})
            for p in papers
        ]
        splits = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(docs)
        vectordb = Chroma.from_documents(splits, HuggingFaceEmbeddings())
        analyzer = ScientificPaperAnalyzer(vectordb.as_retriever(), self.llm)

        return analyzer.analyze(analysis_prompt)

    def _generate_script(self, chain) -> str:
        """Runs the ScriptGenerator and returns the script."""
        from core.generator.script_generator import ScriptGenerator

        gen = ScriptGenerator(chain=chain)
        gen.run()
        return gen.get_final_script()

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        """
        Optional per-row fields:
          • do_extract / do_analyze / do_generate  → bool
          • analysis_prompt   (str)
          • generation_prompt (str)  *not yet used, reserved*
        """
        from core.extract_text.arxiv_search import ArxivSearcher

        rows_out: list[dict] = []

        for ridx, row in model_input.iterrows():
            f_extract = bool(row.get("do_extract", True))
            f_analyze = bool(row.get("do_analyze", True))
            f_generate = bool(row.get("do_generate", True))

            query = row["query"]
            max_results = int(row.get("max_results", 3))
            chunk_size = int(row.get("chunk_size", 1200))
            chunk_overlap = int(row.get("chunk_overlap", 400))

            analysis_prompt = row.get(
                "analysis_prompt",
                "Summarize the content in Portuguese (≈150 words).",
            )

            logging.info(
                "[%d] query='%s' | ext=%s ana=%s gen=%s",
                ridx,
                query,
                f_extract,
                f_analyze,
                f_generate,
            )

            papers: list[dict] = []
            if f_extract:
                t0 = time.perf_counter()
                papers = ArxivSearcher(query, max_results).search_and_extract()
                logging.info(
                    "[%d] %d articles extracted in %.1fs",
                    ridx,
                    len(papers),
                    time.perf_counter() - t0,
                )

            # short-circuit: extraction only
            if f_extract and not f_analyze:
                rows_out.append(
                    {
                        "extracted_papers": json.dumps(papers, ensure_ascii=False),
                        "script": "",
                    }
                )
                continue

            summary = ""
            chain_for_generation = None
            if f_analyze and papers:
                summary = self._analyze_papers(
                    papers, analysis_prompt, chunk_size, chunk_overlap
                )
                # recreate retriever as before (could be cached)
                from core.analyzer.scientific_paper_analyzer import ScientificPaperAnalyzer
                from langchain.schema import Document
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma

                docs = [
                    Document(page_content=p["text"], metadata={"title": p["title"]})
                    for p in papers
                ]
                splits = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ).split_documents(docs)
                vectordb = Chroma.from_documents(splits, HuggingFaceEmbeddings())
                analyzer_tmp = ScientificPaperAnalyzer(
                    vectordb.as_retriever(), self.llm
                )
                chain_for_generation = analyzer_tmp.get_chain()

            # short-circuit: analysis only
            if f_analyze and not f_generate:
                rows_out.append(
                    {
                        "extracted_papers": json.dumps(papers, ensure_ascii=False),
                        "script": summary,
                    }
                )
                continue

            script = ""
            if f_generate and chain_for_generation is not None:
                script = self._generate_script(chain_for_generation)

            rows_out.append(
                {
                    "extracted_papers": json.dumps(papers, ensure_ascii=False),
                    "script": script or summary,
                }
            )

        return pd.DataFrame(rows_out)

    @classmethod
    def log_model(
        cls,
        artifact_path: str = "script_generation_model",
        llm_artifact: str = "models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf",
        config_yaml: str = "configs/config.yaml",
        secrets_yaml: str = "configs/secrets.yaml",
    ):
        core_path, src_path = add_project_dirs_to_syspath()

        artifacts = {
            "config": str(Path(config_yaml).resolve()),
            "secrets": str(Path(secrets_yaml).resolve()),
            "llm": llm_artifact,
        }

        signature = ModelSignature(
            inputs=Schema(
                [
                    ColSpec("string", "query"),
                    ColSpec("integer", "max_results"),
                    ColSpec("integer", "chunk_size"),
                    ColSpec("integer", "chunk_overlap"),
                    ColSpec("boolean", "do_extract"),
                    ColSpec("boolean", "do_analyze"),
                    ColSpec("boolean", "do_generate"),
                    ColSpec("string", "analysis_prompt"),
                    ColSpec("string", "generation_prompt"),
                ]
            ),
            outputs=Schema(
                [
                    ColSpec("string", "extracted_papers"),
                    ColSpec("string", "script"),
                ]
            ),
        )

        code_paths = [str(core_path)]
        if src_path:
            code_paths.append(str(src_path))

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=cls(),
            artifacts=artifacts,
            signature=signature,
            pip_requirements=[
                "mlflow",
                "PyYAML",
                "requests",
                "pymupdf",
            ],
            code_paths=code_paths,
        )



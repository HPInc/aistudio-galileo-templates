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
        values["sampling_params"] = tensorrt_llm.SamplingParams(temperature=0.8, top_p=0.95)
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
        
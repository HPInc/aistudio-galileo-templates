import httpcore
from typing import Any, Dict, List

class LLMMetadataUpdater:
    """
    Class to update metadata (context) of a data structure using a language model.
    
    The class receives in its constructor a chain object (either an API-based model or a local model)
    that must implement the `invoke` method. The `update_context` method iterates over each item in the
    provided data structure, calls the language model to generate a new context based on the code, filename,
    and current context, and updates the item with the returned result.
    
    Example usage:
        from my_module import LLMMetadataUpdater
        # Assume that `llm_chain` is an instance of a model with an `invoke` method
        updater = LLMMetadataUpdater(llm_chain=llm_chain)
        updated_data = updater.update_context(data_structure)
    """

    def __init__(self, llm_chain: Any) -> None:
        """
        :param llm_chain: Object that implements the `invoke` method to call the language model.
        """
        self.llm_chain = llm_chain

    def update_context(self, data_structure: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates the context of each item in the data structure using the language model.
        
        For each item (which must contain the keys 'code', 'filename', and 'context'):
          - Attempts to call the language model to generate an explanation or updated context.
          - In case of connection errors, protocol errors, or other exceptions, retains the original context.
        
        :param data_structure: List of dictionaries containing the data to be processed.
        :return: The updated data structure with new contexts.
        """
        updated_structure: List[Dict[str, Any]] = []

        for item in data_structure:
            code = item.get('code', '')
            filename = item.get('filename', '')
            context = item.get('context', '')
            
            try:
                response = self.llm_chain.invoke({
                    "code": code,
                    "filename": filename,
                    "context": context
                })
                item['context'] = response.strip()

            except httpcore.ConnectError as e:
                print(
                    f"Connection error processing file {filename}: "
                    f"The connection to the API or model has been corrupted. Details: {str(e)}"
                )
                item['context'] = context

            except httpcore.ProtocolError as e:
                print(f"Protocol error when processing the file {filename}: {str(e)}")
                item['context'] = context

            except Exception as e:
                print(f"Error processing the file {filename}: {str(e)}")
                item['context'] = context

            updated_structure.append(item)

        return updated_structure



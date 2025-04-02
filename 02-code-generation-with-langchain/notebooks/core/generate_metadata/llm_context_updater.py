import httpcore

class LLMContextUpdater:
    def __init__(self, llm_chain, verbose=False):
        """
        Initialize the context updater with an LLM chain and verbosity flag.

        :param llm_chain: An LLM chain object that has an .invoke() method
        :param verbose: If True, will print logs for each update or error
        """
        self.llm_chain = llm_chain
        self.verbose = verbose

    def update(self, data_structure):
        """
        Update the context field of each item in the structure using LLM.

        :param data_structure: List of dictionaries with keys: code, filename, context
        :return: Updated data_structure with new context values
        """
        updated_structure = []

        for item in data_structure:
            code = item['code']
            filename = item['filename']
            context = item['context']

            try:
                """""Attempt to generate a new context using the LLM"""""
                response = self.llm_chain.invoke({
                    "code": code,
                    "filename": filename,
                    "context": context
                })

                """""Update context with the LLM's explanation"""""
                item['context'] = response.strip()
                if self.verbose:
                    print(f"[LOG] Context generated for file {filename}: {item['context']}")

            except httpcore.ConnectError as e:
                """""Handle connection issues to the LLM model/API"""""
                if self.verbose:
                    print(f"[ERROR] Connection error processing file {filename}: {str(e)}")
                item['context'] = context

            except httpcore.ProtocolError as e:
                """""Handle protocol errors from the LLM API"""""
                if self.verbose:
                    print(f"[ERROR] Protocol error processing file {filename}: {str(e)}")
                item['context'] = context

            except Exception as e:
                """""Catch any other exception during LLM invocation"""""
                if self.verbose:
                    print(f"[ERROR] General error processing file {filename}: {str(e)}")
                item['context'] = context

            updated_structure.append(item)

        return updated_structure



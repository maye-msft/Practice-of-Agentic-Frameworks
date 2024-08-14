import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

class OAI:
    def __init__(self):
        init_params = {}
        api_type = os.environ.get("OPENAI_API_TYPE")
        if os.getenv("OPENAI_API_VERSION") is not None:
            init_params["api_version"] = os.environ.get("OPENAI_API_VERSION")
        if os.getenv("OPENAI_ORG_ID") is not None:
            init_params["organization"] = os.environ.get("OPENAI_ORG_ID")
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        if os.getenv("OPENAI_API_BASE") is not None or os.getenv("AZURE_OPENAI_ENDPOINT") is not None:
            if api_type == "azure":
                init_params["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            else:
                init_params["base_url"] = os.environ.get("OPENAI_API_BASE")

        init_params["api_key"] = os.environ.get("OPENAI_API_KEY")

        # A few sanity checks
        if api_type == "azure":
            if init_params.get("azure_endpoint") is None:
                raise ValueError(
                    "OPENAI_API_BASE is not set in environment variables, this is required when api_type==azure"
                )
            if init_params.get("api_version") is None:
                raise ValueError(
                    "OPENAI_API_VERSION is not set in environment variables, this is required when api_type==azure"
                )
            if init_params["api_key"].startswith("sk-"):
                raise ValueError(
                    "OPENAI_API_KEY should not start with sk- when api_type==azure, "
                    "are you using openai key by mistake?"
                )
            from openai import AzureOpenAI as Client
        else:
            from openai import OpenAI as Client
        self.client = Client(**init_params)
        
class OAIEmbedding(OAI):

    def generate(self, text: str) -> List[float]:
        return self.client.embeddings.create(
            input=text, model=os.environ.get("EMBEDDING_MODEL_DEPLOYMENT_NAME")
        ).data[0].embedding
        
if __name__ == "__main__":
    oai = OAIEmbedding()
    print(oai.generate("Hello, how are you?"))
        
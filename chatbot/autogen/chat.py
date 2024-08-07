from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent
from promptflow.core import tool

from dotenv import load_dotenv
load_dotenv()

import os

config_list = [
    {
        "model":  os.environ["CHAT_MODEL_DEPLOYMENT_NAME"], 
        "api_type": os.environ["OPENAI_API_TYPE"], 
        "api_version": os.environ["OPENAI_API_VERSION"], 
        "base_url": os.environ["AZURE_OPENAI_ENDPOINT"], 
        "api_key": os.environ["OPENAI_API_KEY"]
    }
]
    
def main():
    
    # Create the agent that uses the LLM.
    assistant = ConversableAgent("agent", llm_config={"config_list": config_list})

    # Create the agent that represents the user in the conversation.
    user_proxy = UserProxyAgent("user", code_execution_config=False)

    # Let the assistant start the conversation.  It will end when the user types exit.
    assistant.initiate_chat(user_proxy, message="How can I help you today?")


if __name__ == "__main__":
    main()
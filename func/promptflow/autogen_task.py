from promptflow.core import tool
from autogen import GroupChat, GroupChatManager, register_function, AssistantAgent, UserProxyAgent

from dotenv import load_dotenv
load_dotenv()

import os

class AutoGenChat:
    def __init__(self) -> None:
        
        self.llm_config = {"config_list": [{
            "model":  os.environ["CHAT_MODEL_DEPLOYMENT_NAME"], 
            "api_type": os.environ["OPENAI_API_TYPE"], 
            "api_version": os.environ["OPENAI_API_VERSION"], 
            "base_url": os.environ["AZURE_OPENAI_ENDPOINT"], 
            "api_key": os.environ["OPENAI_API_KEY"]
        }]}
        
        self.mathAssistant = AssistantAgent(
            name="MathAssistant",
            system_message="""You are a smart assistant, you can help with math problems with the tool provided.
If you solve the problem, return a result and end 'TERMINATE' in new line, when you complete the task successfully.
If you cannot solve the problem, return a message of the reason and end 'TERMINATE' in new line.""",
            description="Math Assistant",
            llm_config=self.llm_config)
        
        self.user_proxy = UserProxyAgent(
            name="User",
            llm_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
            human_input_mode="NEVER"
        )
        
        def multiply(a: int, b: int) -> int:
            return a * b
        
        register_function(
            multiply,
            caller=self.mathAssistant,  # The assistant agent can suggest calls to the calculator.
            executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
            name="multiply",  # By default, the function name is used as the tool name.
            description="A multiply calculation tool",  # A description of the tool.
        )
        
        def plus(a: int, b: int) -> int:
            return a + b
        
        register_function(
            plus,
            caller=self.mathAssistant,  # The assistant agent can suggest calls to the calculator.
            executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
            name="plus",  # By default, the function name is used as the tool name.
            description="A plus calculation tool",  # A description of the tool.
        )
        
        def minus(a: int, b: int) -> int:
            return a - b
        
        register_function(
            minus,
            caller=self.mathAssistant,  # The assistant agent can suggest calls to the calculator.
            executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
            name="minus",  # By default, the function name is used as the tool name.
            description="A minus calculation tool",  # A description of the tool.
        )
        
        def divide(a: int, b: int) -> int:
            return a / b
        
        register_function(
            divide,
            caller=self.mathAssistant,  # The assistant agent can suggest calls to the calculator.
            executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
            name="divide",  # By default, the function name is used as the tool name.
            description="A divide calculation tool",  # A description of the tool.
        )
        
        self.groupchat = GroupChat(agents=[self.user_proxy, self.mathAssistant], messages=[], max_round=12)
        self.manager = GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)

        
    def chat(self, question: str, chat_history: list) -> str:
        res = self.user_proxy.initiate_chat(
                self.manager, message=question,
        )
        return res.summary
    

        
if __name__ == "__main__":
    agent = AutoGenChat()
    print(agent.chat("What is (121 * 3) + 42?"))



agent = AutoGenChat()
        
@tool
def autogen_task(question : str, chat_history : list) -> str:
    answer = agent.chat(question, chat_history)
    return answer
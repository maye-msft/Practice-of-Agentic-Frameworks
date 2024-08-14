
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

from dotenv import load_dotenv
import os

class LangChainMathAgent:
    def __init__(self) -> None:

        @tool
        def add(a: int, b: int) -> int:
            """Adds a and b.

            Args:
                a: first int
                b: second int
            """
            return a + b


        @tool
        def multiply(a: int, b: int) -> int:
            """Multiplies a and b.

            Args:
                a: first int
                b: second int   
            """
            return a * b


        tools = [add, multiply]

        @tool
        def minus(a: int, b: int) -> int:
            """Subtracts b from a.

            Args:
                a: first int
                b: second int
            """
            return a - b

        tools.append(minus)

        @tool
        def divide(a: int, b: int) -> int:
            """Divides a by b.

            Args:
                a: first int
                b: second int
            """
            return a / b
        
        tools.append(divide)


        llm = AzureChatOpenAI(
            deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],        
            openai_api_type=os.environ["OPENAI_API_TYPE"],
            openai_api_version=os.environ["OPENAI_API_VERSION"],
            temperature=0,
            streaming=True
        )

        # Create a MessagesPlaceholder for the chat history
        history_placeholder = MessagesPlaceholder("history")

        # Construct the prompt template
        prompt_template = ChatPromptTemplate.from_messages([    
            ("system", "You are smart agent to answer any question and use tools to do calculation."),
            history_placeholder,
            ("user", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

        agent = create_tool_calling_agent(llm, tools, prompt_template)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
    def execute(self, question: str, chat_history: list) -> str:
        res = self.agent_executor.invoke({"input": question, "history": chat_history})
        return res["output"]
    
    
if __name__ == "__main__":
    agent = LangChainMathAgent()
    response = agent.execute("What is 2 plus 2?", [])
    print(response)
    
   
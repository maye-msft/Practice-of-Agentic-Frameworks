
# Function Invocation

It would be convenient if we can directly provide some functions and let LLM execute them, even though we know that LLM can be used to generate and execute code, 

In the following chapters, we will introduce how to use Promptflow, LangChain, and LlamaIndex to implement function invocation. We hope our chatbot can complete a math question below:

__What is (121 * 3) + 42?__

### Promptflow + AutoGen


We found that Promptflow does not provide the function to automatically recognize and call functions.

So we either implement a node ourselves to determine whether and which function should be called, or we can use another framework called [AutoGen](https://microsoft.github.io/autogen/) to complete this task. And AutoGen can be easily integrated with Promptflow.

> AutoGen offers a unified multi-agent conversation framework as a high-level abstraction of using foundation models. It features capable, customizable and conversable agents which integrate LLMs, tools, and humans via automated agent chat. By automating chat among multiple capable agents, one can easily make them collectively perform tasks autonomously or with human feedback, including tasks that require using tools via code.

AutoGen provides an encapsulation class AssistantAgent to simplify the development of agents. We only need to provide the name, description, system message, and llm configuration to complete a simple agent implementation.

```python
self.mathAssistant = AssistantAgent(
    name="MathAssistant",
    system_message="""You are a smart assistant, you can help with math problems with the tool provided.
If you solve the problem, return a result and end 'TERMINATE' in new line, when you complete the task successfully.
If you cannot solve the problem, return a message of the reason and end 'TERMINATE' in new line.""",
    description="Math Assistant",
    llm_config=self.llm_config)
```

We also need to create a UserProxy Agent, which is mainly used to execute the Python functions we provide in our example.


```python
self.user_proxy = UserProxyAgent(
            name="User",
            llm_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
            human_input_mode="NEVER"
        )
```

Then we can see what Python functions we need to provide to implement a math calculation Agent. We only use addition as an example.


```python
def plus(a: int, b: int) -> int:
    return a + b

register_function(
    plus,
    caller=self.mathAssistant,  # The assistant agent can suggest calls to the calculator.
    executor=self.user_proxy,  # The user proxy agent can execute the calculator calls.
    name="plus",  # By default, the function name is used as the tool name.
    description="A plus calculation tool",  # A description of the tool.
)
```

We define an addition function, and then register it to mathAssistant through the register_function function. That is, mathAssistant will decide when to call this function, and user_proxy will execute this function.

Next, we create a group chat and a group chat manager and put the two agents into the group chat.

```python
self.groupchat = GroupChat(agents=[self.user_proxy, self.mathAssistant], messages=[], max_round=12)
self.manager = GroupChatManager(groupchat=self.groupchat, llm_config=self.llm_config)
```

The chat manager will manage the conversation between the two agents by selecting which agent will handle next message.

Finally, we can start the chat. Here we create a method.

```python
def chat(self, question: str, chat_history: list) -> str:
    res = self.user_proxy.initiate_chat(
            self.manager, message=question,
    )
    return res.summary
```

And this method will be invoked in the Promptflow task node, as below.

```python
agent = AutoGenChat()
        
@tool
def autogen_task(question : str, chat_history : list) -> str:
    answer = agent.chat(question, chat_history)
    return answer
```

> We pass the chat history to the chat method, but we have not used it in this example. We can use it to record the conversation history and provide context for the conversation.

Here is the command to start this chatbot.

```shell
# Create a .env file and replace the placeholders with your own values
cp  .env.example .env
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install dependencies
cd func/promptflow
# Install dependencies
pip install -r requirements.txt
# Run the flow locally
pf flow test --flow . --interactive --ui
```


Here is a screenshot of the UI.

![promptflow autogen ui](../images/promptflow-autigen-math.png)

You can find the complete code example in this [directory](./func/promptflow/).

### LangChain

Here is the sample to implement the math calculation agent using LangChain.

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b
tools.append(add)
```

We need to define the add function and use the @tool decorator to register it as a tool. The tool is a mechanism in LangChain to manage the functions that can be called by the LLM model, where the docstring of the function will be used as the description of the tool.

The statement 'tools.append(add)' is used to append the function to the tools list.

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent

agent = create_tool_calling_agent(llm, tools, prompt_template)
self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

The code above shows how to create an agent with the tools defined. The agent_executor is used to execute the agent.

```python
def execute(self, question: str, chat_history: list) -> str:
  res = self.agent_executor.invoke({"input": question, "history": chat_history})
  return res["output"]
```

We provide a method to execute the agent. The method will be invoked by external code.

We use [Chainlit](https://github.com/Chainlit/chainlit) as the front-end to interact with the LangChain agent.

```python
import chainlit as cl
from agent import LangChainMathAgent

@cl.on_chat_start
async def on_chat_start():
    agent = LangChainMathAgent()
    cl.user_session.set("prompt_history",[])
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: LangChainMathAgent
    prompt_history = cl.user_session.get("prompt_history")  # type: list
    res = agent.execute(message.content, prompt_history)
    prompt_history.append(("user", message.content))
    prompt_history.append(("system", res))
    await cl.Message(content=res).send()
```

Chainlit provide a pure Python way to interact with the LangChain agent. We initialize the agent and the prompt history in the on_chat_start function. And we use the on_message function to handle the message from the user.

```shell
chainlit run app.py -w
```
Here is the screenshot of the UI.

![langchain math ui](../images/langchain-math.png)

You can find the complete code example in this [directory](./func/langchain/).

### LlamaIndex

We can also implement the math calculation agent using LlamaIndex.

```python
from llama_index.core.tools import FunctionTool

def add(a: int, b: int) -> int:
    """Add two integers and returns the result integer"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

self.agent = ReActAgent.from_tools([multiply_tool, add_tool, minus_tool, divide_tool], llm=llm, verbose=True)
```

From the code above, we can see that we define an add function and then leverage the FunctionTool provided by LlamaIndex to create a tool. We can also create multiply_tool, minus_tool, and divide_tool in the same way.

Then we create an agent with the tools we defined.

```python
def execute(self, question: str) -> str:
    response = self.agent.chat(question)
    return response.response
```

We provide a method to execute the agent. The method will be invoked by Chainlit or Promptflow.

Here is a sample code of Chainlit to integrate with LlamaIndex.

```python
import chainlit as cl
from agent import LlamaIndexMathAgent

@cl.on_chat_start
async def on_chat_start():
    agent = LlamaIndexMathAgent()
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: LlamaIndexMathAgent
    res = agent.execute(message.content)
    await cl.Message(content=res).send()
```

Here is the screenshot of the UI.

![llamaindex math ui](./images/llamaindex-math.png)

> We found LlamaIndex supports chat history by default. 

You can find the complete code example in this [directory](./func/llamaindex/).
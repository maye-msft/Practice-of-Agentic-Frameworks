# Practices of Agentic Frameworks

## Introduction

AI Agent is an intelligent agent that uses large language models, combined with existing knowledge bases, APIs, and the coding capabilities of large language models. It has been proven that these intelligent agents can help us complete some complex tasks.

With the popularity of intelligent agents, in order to simplify their development, some open-source frameworks have emerged in the open-source community. Here we introduce some of the open-source frameworks and provide examples of common scenarios.


- [Promptflow](https://microsoft.github.io/promptflow/)
    > Prompt flow is a suite of development tools designed to streamline the end-to-end development cycle of LLM-based AI applications, from ideation, prototyping, testing, evaluation to production deployment and monitoring. It makes prompt engineering much easier and enables you to build LLM apps with production quality.

- [LangChain](https://python.langchain.com/v0.2/docs/introduction/)
    > LangChain is a framework designed to simplify the creation of applications using large language models (LLMs). As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis.

- [LlamaIndex](https://docs.llamaindex.ai/en/stable/)
    > LlamaIndex is a framework for building context-augmented generative AI applications with LLMs.

 We will demonstrate the development practices of these frameworks in 
 - Chatbot
 - Agent Implementation
 - Workflow orchestration
 - Retrieval-Augmented Generation(RAG)


## Chatbot

### Promptflow

Here is [an example](https://github.com/microsoft/promptflow/tree/main/examples/flows/chat/chat-basic) of a basic chatbot using Promptflow. 


We can see that the application of Promptflow orchestrates the conversation flow through a yaml file. In this example, it contains two input items, the history of the conversation and the question, and an output item, which is the answer to the question.

```yaml
inputs:
  chat_history:
    type: list
    default: []
  question:
    type: string
    is_chat_input: true
    default: What is ChatGPT?
outputs:
  answer:
    type: string
    reference: ${chat.output}
    is_chat_output: true
```

It is an simple example, the whole flow only contains one node, which is a chat node. We name this node as 'chat', and the type is 'llm'.


The input of this node contains the parameters related to the LLM model, such as the model name, the maximum number of tokens, the temperature, etc., as well as the chat history and question passed from the previous node, which is the entrance of the conversation.


The source represents how the prompt of this node is generated, which is generated through a jinja2 template file here. Combined with the content of the yaml file, we can see how the template file generates the conversation. The jinja template uses two variables, chat_history and question, to generate the prompt.

```yaml
  inputs:
    deployment_name: gpt-4o
    model: gpt-4o
    max_tokens: "1024"
    temperature: "0.7"
    chat_history: ${inputs.chat_history}
    question: ${inputs.question}
  name: chat
  type: llm
  source:
    type: code

```yaml
  inputs:
    deployment_name: gpt-4o
    model: gpt-4o
    max_tokens: "1024"
    temperature: "0.7"
    chat_history: ${inputs.chat_history}
    question: ${inputs.question}
  name: chat
  type: llm
  source:
    type: code
    path: chat.jinja2
  api: chat
  connection: open_ai_connection
```

jinjia2 template

```jinja2
# system:
You are a helpful assistant.

{% for item in chat_history %}
# user:
{{item.inputs.question}}
# assistant:
{{item.outputs.answer}}
{% endfor %}

# user:
{{question}}
```

In this example, we use a connection configuration named open_ai_connection to configure the OpenAI link information.

In the local development environment, we can create it through the command-line tool provided by Promptflow.

Here is an example of creating a connection configuration with azure openai.

```shell
# Override keys with --set to avoid yaml file changes
pf connection create --file ../../connections/azure_openai.yml --set api_key=<your_api_key> api_base=<your_api_base> --name open_ai_connection
```

An example to create a connection configuration with openai.

```shell
# Override keys with --set to avoid yaml file changes
pf connection create --file ../../connections/openai.yml --set api_key=<your_api_key> --name open_ai_connection
```


The above command creates a connection configuration named open_ai_connection through a file named azure_openai.yml and our api_key and api_base.

```yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/AzureOpenAIConnection.schema.json
name: open_ai_connection
type: azure_open_ai
api_key: "<user-input>"
api_base: "aoai-api-endpoint"
api_type: "azure"
```


We also need a requirements.txt file to specify the dependencies.

```txt
promptflow
promptflow-tools
```

You can find the complete code example in this [directory](./chatbot/promptflow/).


Here is the complete command to start this chatbot.

```shell
# Create a virtual environment and install dependencies
python3 -m venv .venv
# Activate the virtual environment
source .venv/bin/activate
# Install dependencies
cd chatbot/promptflow
pip install -r requirements.txt
# Create the connection
pf connection create --file ../../connections/azure_openai.yml --set api_key=<your_api_key>  api_base=<your_api_base> --name open_ai_connection
# Run the flow locally
pf flow test --flow . --interactive --ui
```

You can open the URL show in the console with your browser to interact with the chatbot.

![promptflow chatbot ui](./images/promptflow-chatbot-ui.png)


### LangChain

Compared with Promptflow, AutoGen is more code-oriented. In this example, we can see that the chatbot is implemented in a Python script. It looks like simpler than Promptflow. But it cna only run in terminal.

```python
llm = AzureChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],     
    openai_api_type=os.environ["OPENAI_API_TYPE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],  
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"], 
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are smart agent to answer any question."),
        ("user", "{input}"),
    ]
)

chain = LLMChain(llm=llm, prompt=prompt, output_key="metrics")
res = chain({"input": "What is ChatGPT?"})
print(res["metrics"])
```

It initializes an AzureChatOpenAI object, which is a wrapper of the Azure OpenAI API. From this example, we can also find LangChain has good support for prompt generation. LLMChain is a class that can be used to chain the LLM model and the prompt together. 

And here is an experiment to integrate the LangChain chatbot with Promptflow, so that we can leverage Promptflow UI to run LangChain application.

Here we use the Promptflow tool to make the LangChain chatbot as a task of the flow. You can find more information about the Promptflow tool in later chapters.

```python
# Create a MessagesPlaceholder for the chat history
history_placeholder = MessagesPlaceholder("history")

# Construct the prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are smart agent to answer any question."),
    history_placeholder,
    ("user", "{input}")
])
    
@tool
def langhcian_task(question : str, chat_history : list) -> str:
    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="metrics")
    res = chain({"input": question, "history": format_chat_history(chat_history)})
    return res["metrics"]
    
def format_chat_history(chat_history):
    formatted_chat_history = []
    for message in chat_history:
        if "inputs" in message:
            formatted_chat_history.append(("user", message["inputs"]["question"]))
        if "outputs" in message:
            formatted_chat_history.append(("system", message["outputs"]["answer"]))
    return formatted_chat_history
```

And the flow.dag.yml file is like this.

```yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
environment:
  python_requirements_txt: requirements.txt
inputs:
  chat_history:
    type: list
    default: []
  question:
    type: string
    is_chat_input: true
    default: What is ChatGPT?
outputs:
  answer:
    type: string
    reference: ${langchain_task.output}
    is_chat_output: true
nodes:
- name: langchain_task
  type: python
  source:
    type: code
    path: langchain_task.py
  inputs:
    question: ${inputs.question}
    chat_history: ${inputs.chat_history}
```

You can find the complete code example in this [directory](./chatbot/langchain/).

Here is the command to start this chatbot which is the same as the Promptflow chatbot.

```shell
# Run the flow locally
pf flow test --flow . --interactive --ui
```

### LlamaIndex

The chatbot based on LlamaIndex is also implemented in a Python script. It is similar to LangChain.

```python
llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)

chat_engine = SimpleChatEngine.from_defaults(llm=llm)

response = chat_engine.chat(
    "What is ChatGPT?"
)
print(response)
```

And we can also integrate the LlamaIndex chatbot with Promptflow.

```python
llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name=os.environ["CHAT_MODEL_DEPLOYMENT_NAME"],
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["OPENAI_API_VERSION"],
)
    
chat_engine = SimpleChatEngine.from_defaults(llm=llm)

@tool
def llama_task(question : str) -> str:
    response = chat_engine.chat(
        question
    )
    return response.response
```

And the flow.dag.yml file is like this.

```yaml
$schema: https://azuremlschemas.azureedge.net/promptflow/latest/Flow.schema.json
environment:
  python_requirements_txt: requirements.txt
inputs:
  question:
    type: string
    is_chat_input: true
    default: What is ChatGPT?
outputs:
  answer:
    type: string
    reference: ${llamaindex_task.output}
    is_chat_output: true
nodes:
- name: llamaindex_task
  type: python
  source:
    type: code
    path: llamaindex_task.py
  inputs:
    question: ${inputs.question}
```

Here is the command to start this chatbot which is the same as the Promptflow and LangChain chatbot.

```shell
# Run the flow locally
pf flow test --flow . --interactive --ui
```

You may find the chatbot based on LlamaIndex supports chat history by default.
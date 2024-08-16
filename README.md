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
 - [Chatbot](./chatbot/README.md):
It demonstrates the development practices of these frameworks to build a chatbot with Azure OpenAI API.

 - [Function Invocation](./func/README.md):
It demonstrates how to create agent and invoke functions provided by users, where the agent can support math calculations.

 - [Retrieval-Augmented Generation(RAG)](./rag/README.md):
It demonstrates how to build a chatbot with RAG which uses context information from a document to answer questions.












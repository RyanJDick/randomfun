# Document QA

A simple / minimal `doc_qa` tool for question answering about PDF documents.

## Setup

- Tested with `python 3.8.6`.
- Setup python virtual environment, and install deps with `pip install -r requirements.txt`.
- Install the `doc_qa` module: `pip install -e .`
- Set the `OPENAI_API_KEY` environment variable.

## Getting Started

```bash
python ask.py <path/to/pdf/document>

# Enter your query when prompted.
# (The tool supports chat-based interaction (it has the context from your previous queries).)
```
## Sample Interaction

There is a sample interaction with the doc_qa agent [here](SAMPLE_DOC_QA.md).

## Architecture

This section describes the high-level architecture for the document question answering agent implemented in this repo.

This architecture was designed with **simplicity** as a top priority. There are many aspects of this architecture that would change based on the requirements of a particular application (cost sensitivity, document type, document length, query type, correctness sensitivity, etc.).

There are two core components to the architecture: the document embedding vector store, and the chat agent.

### Embedding Vector Store

The vector store must first be initialized with the contents of the document of interest. The initialization process is as follows:
1. The user provides a path to a PDF document of interest.
2. The PDF document is parsed into a text representation. (Any visual/graphical information is lost at this stage.)
3. The PDF text is split into small chunks. Chunks are approximately 1000 characters long, with some effort made to split chunks at logical delimiters (e.g. newline characters).
4. Each chunk is converted to an embedding vector using [OpenAI's embedding API](https://platform.openai.com/docs/api-reference/embeddings).
5. The embedding vectors are added to a vector store that enables fast retrieval based on embedding similarity (cosine similarity).

Later, information can be retrieved from the vector store as follows:
1. A query is provided that describes the content to be retrieved from the vector store. In this implementation, the query is generated by an LLM agent, but in other applications it could be written by a human.
2. The query is converted to an embedding vector using the **same** embedding API that was used to initialize the vector store.
3. The query embedding vector is used to retrieve the text chunks (aka "documents") from the original document with the greatest 'similarity' to the query. There are some options in how the similarity/retrieval algorithm is implemented. In this implementation we simply return the `N` chunks with the greatest embedding vector cosine similarity.

### Chat Agent

The chat agent is built on top of [OpenAI's completion API](https://platform.openai.com/docs/api-reference/completions).

The key to achieving the desired document question answering behaviour is the initial model prompt. The full prompt can be found in [this file](doc_qa/agent/prompt.py). The prompt has the important key components:
- The model is given the context that it should behave as a document question answering agent, and that it should assume all questions are about the document of interest.
- The model is given plain text instructions for how it can use the embedding vector store as a tool to query for information from the document.
- The model is encouraged to not make things up and to only give answers when it has high confidence. (This part still needs some improvement.)

The chat agent calls the OpenAI completion API repeatedly. Each call contains either new human input or results from a query to the document embedding query store. In order to achieve a chat-like interface, calls to the completion API always include the full history of previous interactions.

## Future Work

- Experiment with using OpenAI's 'chat' API rather than the 'completion' API. The chat-optimized models will likely provide a better chat experience. Also, the initial [system prompt](https://platform.openai.com/docs/guides/chat/introduction) to the chat API is supposed to be more heavily weighted, which could be beneficial for this use case.
- The agent occasionally makes things up rather than basing its responses on the document. Here's an example failure case from a question about the [SAM paper](https://arxiv.org/pdf/2304.02643.pdf):
```bash
Enter a query: What does SAM stand for?

> Entering new AgentExecutor chain...

Thought: Do I need to use a tool? No
AI: SAM stands for Security Access Manager.

> Finished chain.
**********************************
```
- The agent sometimes fails to find the relevant text from the document when querying the vector store. For example, asking for the authors of the paper often returns authors from works cited in the paper rather than the paper itself.
- Currently there is zero overlap between text chunks in the vector store. Some chunk overlap could be a good idea.
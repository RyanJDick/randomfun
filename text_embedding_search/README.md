# Text Embedding Search

Experiments related to natural language search powered by LLM text embeddings (e.g. document question answering, information retrieval).

## Setup

- Tested with `python 3.8.6`.
- Setup python virtual environment, and install deps with `pip install -r requirements.txt`.
- Install the `doc_qa` module: `pip install -e .`
- Set the `OPENAI_API_KEY` environment variable.

## Getting Started

```
python ask.py <path/to/pdf/document>
```

## TODO

Problems
- Model makes stuff up based on prior knowledge i.e. not based on the document

- How well does information retrieval from embeddings work?
  - Is writing a query an effective way of finding relevant text?
- Distinction: summarization vs information retrieval.

- Possible architecture:
  - Retrieval tool strictly returns N closest embedding matches (No Q/A aspect to it)

- 

## Observations

- Make sure you give the agent enough of the document. Just giving it small chunks retrieved based on embeddings does not seem to work well.


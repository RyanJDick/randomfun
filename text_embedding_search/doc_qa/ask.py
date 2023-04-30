import argparse

from langchain.agents import Tool
from langchain.agents.agent import AgentExecutor
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.base import VectorStore

from doc_qa.agent import DocQAChatAgent
from doc_qa.embed import generate_pdf_embeddings


def build_retrieve_fn(vectorstore: VectorStore, num_docs=4):
    def retrieve_fn(query):
        results = vectorstore.similarity_search(query, k=num_docs)

        s = ""
        for i, res in enumerate(results):
            s += f"Block {i} content:\n"
            s += "---\n"
            s += res.page_content
            s += "\n---\n\n"

        return s

    return retrieve_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="Path to PDF file to ask questions about.")
    args = parser.parse_args()

    # TODO: Experiment with using a chat-optimized model instead (e.g. GPT-3 or GPT-4).
    llm = OpenAI(temperature=0)

    vector_store = generate_pdf_embeddings(args.pdf_file)

    tools = [
        Tool(
            name="DOC_RETRIEVER",
            func=build_retrieve_fn(vector_store),
            description="DOC_RETRIEVER returns blocks of text from a document that are relevant to the input query. The input should be a statement describing the content that you want to retrieve from the document.",
        ),
    ]

    agent = DocQAChatAgent.from_llm_and_tools(llm=llm, tools=tools)

    memory = ConversationBufferMemory(memory_key="chat_history")

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        # return_intermediate_steps=True,
        memory=memory,
        # callback_manager=callback_manager,
        # **kwargs,
    )

    while True:
        print("**********************************\n\n")
        query = input("Enter a query: ")
        agent_executor.run(query)


if __name__ == "__main__":
    main()

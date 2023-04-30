import argparse

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
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

    # embedding_llm = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)

    index = generate_pdf_embeddings(args.pdf_file)

    doc_retrieval = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
    )

    tools = [
        Tool(
            name="DOC_RETRIEVER",
            func=build_retrieve_fn(index.vectorstore),  # doc_retrieval.run,
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

        result = agent_executor.run(query)
        # result = agent_executor({"chat_history": chat_history, "input": query})
        # result = index.query_with_sources(query)

        # print(result)

        # import json

        # print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import argparse

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent import AgentExecutor
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from doc_qa.agent import DocQAChatAgent
from doc_qa.embed import generate_pdf_embeddings


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
            func=doc_retrieval.run,
            description="useful for retrieving information from the document of"
            " interest. Input should be a fully-formed question.",
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

import argparse

from doc_qa.embed import generate_pdf_embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_file", help="Path to PDF file to ask questions about.")
    args = parser.parse_args()

    index = generate_pdf_embeddings(args.pdf_file)

    query = input("Enter a query:")

    result = index.query_with_sources(query)

    print(result)


if __name__ == "__main__":
    main()

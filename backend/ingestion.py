import os

from dotenv import load_dotenv

## Load env variable
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


def ingest_docs():
    ##https://python.langchain.com/docs/integrations/document_loaders/csv/ official doc
    loader = CSVLoader(file_path="./backend/item_data/testdata.csv",source_column="item_id")
    raw_documents = loader.load()

    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)

    documents = text_splitter.split_documents(raw_documents)

    # for doc in documents:
    #     new_url = doc.metadata["source"]
    #     new_url = new_url.replace("langchain-docs","https:/")
    #     doc.metadata.update({"source": new_url})

    print(f"Going to add{len(documents)} to pinecone")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "document-reader"

    # Get the index
    index = pc.Index(index_name)

    # ✅ Delete all existing vectors
    #index.delete(delete_all=True)


    PineconeVectorStore.from_documents(
        documents=documents,embedding=embeddings,index_name="document-reader"
    )
    print("************* Loading to vectorstore done *************")
    return True

# if __name__ == "__main__":
#     ingest_docs()
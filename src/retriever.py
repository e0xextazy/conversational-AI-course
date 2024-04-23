from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from tqdm import tqdm


def create_retriever(_docs, retriever_name='retriever', k=5, child_chunk_size=300, parent_chunk_size=1200, model_name_="paraphrase-multilingual-mpnet-base-v2"):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size)
    vectorstore = Chroma(
        collection_name=retriever_name,
        embedding_function=SentenceTransformerEmbeddings(model_name=model_name_)
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": k}
    )

    for doc in tqdm(_docs, desc="Adding documents"):
        retriever.add_documents([doc])

    return retriever
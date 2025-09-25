from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_chroma import Chroma

def prepare_vectorstore(pdf_paths,persist_directory='db',collection_name='articles'):
    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)

    #Embeddings and Text splitter (for this emdedding model chunk size = 700 and overlap = 150-200 is optimal)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap=150)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name)

    #Preprocesse documents
    documents = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDf not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        doc = text_splitter.split_documents(pages)
        
        #Add document to VS and save the state
        vectorstore.add_documents(doc)
        vectorstore.persist()

        #Save documet's metadate
        documents.append({
            "filename": os.path.basename(pdf_path),
            "pages": len(pages),
            "chunks": len(doc)
        }) 

    return vectorstore, documents
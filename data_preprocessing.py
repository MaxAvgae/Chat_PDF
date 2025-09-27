from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
import glob
from langchain_chroma import Chroma

def prepare_vectorstore(pdf_paths, persist_directory='db', collection_name='articles'):
    """Создает или обновляет векторное хранилище с документами"""
    if not os.path.exists(persist_directory):
        os.mkdir(persist_directory)

    # Embeddings and Text splitter (for this embedding model chunk size = 700 and overlap = 150-200 is optimal)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name)

    # Preprocess documents
    documents = []
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        doc = text_splitter.split_documents(pages)
        
        # Add document to VS and save the state
        vectorstore.add_documents(doc)

        # Save document's metadata
        documents.append({
            "filename": os.path.basename(pdf_path),
            "pages": len(pages),
            "chunks": len(doc)
        }) 

    return vectorstore, documents

def load_existing_vectorstore(persist_directory, collection_name):
    """Загружает существующее векторное хранилище"""
    try:
        if not os.path.exists(persist_directory):
            return None
            
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Проверяем, есть ли документы в коллекции
        if vectorstore._collection.count() == 0:
            return None
            
        return vectorstore
    except Exception as e:
        print(f"Ошибка при загрузке векторного хранилища: {e}")
        return None

def get_available_collections(persist_directory):
    """Получает список доступных коллекций в директории"""
    collections = []
    try:
        if not os.path.exists(persist_directory):
            return collections
            
        # Ищем файлы метаданных коллекций
        metadata_files = glob.glob(os.path.join(persist_directory, "*.json"))
        for metadata_file in metadata_files:
            collection_name = os.path.splitext(os.path.basename(metadata_file))[0]
            collections.append(collection_name)
            
        # Также проверяем директории с коллекциями
        for item in os.listdir(persist_directory):
            item_path = os.path.join(persist_directory, item)
            if os.path.isdir(item_path):
                # Проверяем, есть ли файлы Chroma в директории
                chroma_files = glob.glob(os.path.join(item_path, "*.parquet"))
                if chroma_files:
                    collections.append(item)
                    
    except Exception as e:
        print(f"Ошибка при получении списка коллекций: {e}")
        
    return list(set(collections))  # Убираем дубликаты
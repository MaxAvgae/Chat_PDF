import streamlit as st
import os
import json
from data_preprocessing import prepare_vectorstore, load_existing_vectorstore, get_available_collections
from langchain_chroma import Chroma
from agent import create_agent

st.set_page_config(page_title="📚 RAG Agent", page_icon="🤖")
st.title("📚 RAG-агент с управлением базами знаний")

persist_dir = "chroma_db"
collection_name = "statistics"

# Хранилище для загруженных файлов
uploaded_dir = "uploaded_files"
if not os.path.exists(uploaded_dir):
    os.makedirs(uploaded_dir)

# Создаем директорию для баз данных если не существует
if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

# Инициализация session state
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "current_collection" not in st.session_state:
    st.session_state["current_collection"] = None

# Боковая панель для управления базами знаний
with st.sidebar:
    st.header("🗄️ Управление базами знаний")
    
    # Получаем список доступных коллекций
    available_collections = get_available_collections(persist_dir)
    
    if available_collections:
        st.subheader("📋 Существующие базы знаний:")
        for collection in available_collections:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"• {collection}")
            with col2:
                if st.button("Выбрать", key=f"select_{collection}"):
                    st.session_state["current_collection"] = collection
                    st.session_state["agent"] = None  # Сброс агента
                    st.rerun()
        
        st.divider()
    
    # Создание новой базы знаний
    st.subheader("➕ Создать новую базу знаний")
    new_collection_name = st.text_input("Название коллекции:", value="new_collection")
    
    if st.button("Создать новую коллекцию"):
        if new_collection_name and new_collection_name not in available_collections:
            st.session_state["current_collection"] = new_collection_name
            st.session_state["agent"] = None
            st.success(f"Создана новая коллекция: {new_collection_name}")
            st.rerun()
        else:
            st.error("Коллекция с таким именем уже существует или имя не указано")

# Основной интерфейс
if st.session_state["current_collection"]:
    st.subheader(f"📚 Работа с коллекцией: {st.session_state['current_collection']}")
    
    # Загрузка файлов для текущей коллекции
    uploaded_files = st.file_uploader("Загрузи PDF файлы", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(uploaded_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(path)

        st.success(f"Файлы сохранены: {[os.path.basename(p) for p in file_paths]}")

        # Кнопка для создания/обновления векторной БД
        if st.button("Создать/Обновить векторную БД"):
            with st.spinner("Обрабатываю документы..."):
                try:
                    vectorstore, documents = prepare_vectorstore(
                        file_paths, 
                        persist_dir, 
                        st.session_state["current_collection"]
                    )
                    st.success("Векторная база создана/обновлена ✅")
                    
                    # Создаем агента
                    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                    st.session_state["agent"] = create_agent(retriever)
                    
                except Exception as e:
                    st.error(f"Ошибка при создании базы данных: {str(e)}")
    
    # Загрузка существующей базы знаний
    if st.button("Загрузить существующую базу знаний"):
        try:
            vectorstore = load_existing_vectorstore(persist_dir, st.session_state["current_collection"])
            if vectorstore:
                retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
                st.session_state["agent"] = create_agent(retriever)
                st.success("База знаний загружена ✅")
            else:
                st.warning("База знаний пуста или не существует")
        except Exception as e:
            st.error(f"Ошибка при загрузке базы данных: {str(e)}")

# Чат с агентом
if st.session_state["agent"]:
    st.subheader("💬 Задай вопрос агенту")
    user_input = st.chat_input("Напиши вопрос...")
    
    if user_input:
        try:
            result = st.session_state["agent"].invoke({"messages": [{"role": "user", "content": user_input}]})
            answer = result["messages"][-1].content
            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"Ошибка при обработке запроса: {str(e)}")
else:
    st.info("👆 Выберите или создайте базу знаний для начала работы")

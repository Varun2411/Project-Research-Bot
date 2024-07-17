import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables (especially OpenAI API key)
load_dotenv()

# Streamlit UI setup
st.title("Research Bot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

def process_urls(urls):
    try:
        # Load data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading... Started... ✅✅✅")
        data = loader.load()
        if not data:
            main_placeholder.text("No data loaded. Please check the URLs.")
            return
        main_placeholder.text(f"Data Loading... Completed. Loaded {len(data)} documents.")

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting... Started... ✅✅✅")
        docs = text_splitter.split_documents(data)
        if not docs:
            main_placeholder.text("No documents split. Please check the data.")
            return
        main_placeholder.text(f"Text Splitting... Completed. Split into {len(docs)} chunks.")

        # Create embeddings and save to FAISS index
        embeddings = OpenAIEmbeddings()
        main_placeholder.text("Creating Embeddings... Started... ✅✅✅")
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        if not vectorstore_openai:
            main_placeholder.text("FAISS index creation failed.")
            return
        main_placeholder.text("Creating Embeddings... Completed.")

        # Save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        main_placeholder.text("Process Completed Successfully! 🎉")
        
    except Exception as e:
        main_placeholder.text(f"Error during processing: {e}")

if process_url_clicked and urls:
    process_urls(urls)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
                result = chain({"question": query}, return_only_outputs=True)
                
                st.header("Answer")
                st.write(result["answer"])

                sources = result.get("sources", "")
                if sources:
                    st.subheader("Sources:")
                    sources_list = sources.split("\n")
                    for source in sources_list:
                        st.write(source)
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
    else:
        st.error("FAISS index file not found. Please process the URLs first.")


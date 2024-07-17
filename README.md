
## ResearchBot: News Research Tool 📈

It is a Streamlit-based web application designed to help users process and analyze news articles from URLs.
Leveraging the power of OpenAI's language models and FAISS for efficient text retrieval, Bot provides a user-friendly
interface for ingesting, processing, and querying information from news articles. 
This tool is particularly useful for researchers, journalists, and anyone interested in extracting insights from multiple sources of information.

# Features

URL Processing: Enter up to three news article URLs in the sidebar for processing.

Data Loading and Splitting: Loads data from the provided URLs and splits the text into manageable chunks for analysis.

Embeddings Creation: Generates embeddings using OpenAI's language model and stores them in a FAISS index.

Question Answering: Allows users to input queries and retrieves answers along with sources from the processed data.

Persistent Storage: Saves the FAISS index to a file for persistent storage and future queries.

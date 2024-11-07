from langchain.document_loaders import WebBaseLoader
import bs4
import re
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.vectorstores import FAISS

from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

class GradingRetriever:
    def __init__(self, bs_kwargs=None):
        """Initialize the Retriever with an empty list of documents and splits."""
        self.documents = [] 
        self.splits = []  
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300,
            chunk_overlap=50)
        self.ensemble_retriever = None

        self.bs_kwargs = bs_kwargs if bs_kwargs else {
            'parse_only': bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
        }

    def add_documents(self, web_paths):
        """Load new documents from the provided web URLs and add them to the splits."""
        loader = WebBaseLoader(
            web_paths=tuple(web_paths),
            bs_kwargs=self.bs_kwargs
        )
        try:
            # Load the documents from the web
            new_docs = loader.load()
            # Add to the document list
            self.documents.extend(new_docs)
            
            # Split documents and add splits to the splits list
            doc_splits = self.text_splitter.split_documents(new_docs)
            self.splits.extend(doc_splits)
            faiss_index = FAISS.from_documents(self.splits, embedding=OpenAIEmbeddings())
            faiss_retriever = faiss_index.as_retriever()


            bm25_retriever = BM25Retriever.from_documents(self.splits)

            self.ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.4, 0.6])
            return len(new_docs) 
        except Exception as e:
            return f"Error loading documents: {str(e)}"

    def get_splits(self):
        """Return all the document splits."""
        return self.splits
    
    def invoke_query(self, query):
        if self.ensemble_retriever:
            return self.ensemble_retriever.invoke(query)
        return ""

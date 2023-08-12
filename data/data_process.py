from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from utils.custom_logger import logger

class DataLoadPDF:
    """
    A class for loading data from a PDF file.
    """

    def __init__(self, file_path):
        """
        Initialize the DataLoadPDF instance.

        Args:
            file_path (str): Path to the PDF file to load.
        """
        self.file_path = file_path

    def load_data(self):
        """
        Load data from the PDF file.

        Returns:
            list: List of pages from the PDF.
        """
        logger.info(f"Reading file {os.path.basename(self.file_path)} ... ")
        loader = PyPDFLoader(self.file_path)
        pages = loader.load()
        return pages

class DataSplitter:
    """
    A class for splitting data into chunks.
    """

    def __init__(self, chunk_size, chunk_overlap):
        """
        Initialize the DataSplitter instance.

        Args:
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_data(self, pages):
        """
        Split data into chunks.

        Args:
            pages (list): List of data pages.

        Returns:
            list: List of split documents.
        """
        logger.info(f"Document splitting with chunk_size {self.chunk_size} and chunk_overlap {self.chunk_overlap} ... ")
        text_splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        docs = text_splitter.split_documents(pages)
        return docs

class EmbeddingManager:
    """
    A class for managing document embeddings.
    """

    def __init__(self, model_name):
        """
        Initialize the EmbeddingManager instance.

        Args:
            model_name (str): Name of the embedding model.
        """
        self.model_name = model_name
        logger.info(f"Loading embeddings Model {self.model_name} ... ")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)

    def create_embeddings(self, docs):
        """
        Create embeddings for documents.

        Args:
            docs (list): List of documents.

        Returns:
            FAISS: Document embeddings.
        """
        logger.info(f"Creating document embeddings for {len(docs)} split ... ")
        self.doc_embedding = FAISS.from_documents(docs, self.embeddings)
        return self.doc_embedding

    def save_embedding(self, file_name):
        """
        Save document embeddings to a file.

        Args:
            file_name (str): Name of the file to save the embeddings.
        """
        emedding_dir = "embeddings_data"
        if not os.path.exists(emedding_dir):
            os.mkdir(emedding_dir)
        file_name = os.path.basename(file_name)
        logger.info(f"Saving document embeddings: {'embeddings_data/'+file_name} ... ")
        with open("embeddings_data/"+file_name+".pkl", "wb") as f:
            pickle.dump(self.doc_embedding, f)

    def load_embedding(self, file_name):
        """
        Load document embeddings from a file.

        Args:
            file_name (str): Name of the file to load the embeddings.

        Returns:
            FAISS: Loaded document embeddings.
        """
        file_name = os.path.basename(file_name)
        logger.info(f"Loading document embeddings locally: {'embeddings_data/'+file_name} ... ")
        with open("embeddings_data/"+file_name+".pkl", "rb") as f:
            self.doc_embedding = pickle.load(f)
        return self.doc_embedding

    def check_embedding_available(self, file_name):
        """
        Check if document embeddings are available in a file.

        Args:
            file_name (str): Name of the file to check.

        Returns:
            bool: True if document embeddings are available, False otherwise.
        """
        file_name = os.path.basename(file_name)
        doc_check = os.path.isfile("embeddings_data/"+file_name+".pkl")
        logger.info(f"Is document embedding found: {doc_check}")
        return doc_check

class DocumentProcessor:
    """
    A class for processing documents and managing embeddings.
    """

    def __init__(self, model_name, chunk_size, chunk_overlap):
        """
        Initialize the DocumentProcessor instance.

        Args:
            model_name (str): Name of the embedding model.
            chunk_size (int): Size of each chunk.
            chunk_overlap (int): Overlap between consecutive chunks.
        """
        logger.info(f"Initializing document processor parameters - embedding model_name: {model_name}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap} ... ")
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_manager = EmbeddingManager(model_name)

    def process_document(self, file_path):
        """
        Process a document and manage embeddings.

        Args:
            file_path (str): Path to the document file.

        Returns:
            FAISS: Document embeddings.
        """
        if self.embedding_manager.check_embedding_available(file_path):
            return self.embedding_manager.load_embedding(file_path)
        else:
            data_loader = DataLoadPDF(file_path)
            pages = data_loader.load_data()

            data_splitter = DataSplitter(self.chunk_size, self.chunk_overlap)
            docs = data_splitter.split_data(pages)

            doc_embedding = self.embedding_manager.create_embeddings(docs)
            self.embedding_manager.save_embedding(file_path)
            return doc_embedding

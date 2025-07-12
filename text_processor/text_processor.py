"""
GPT Text Processor with RAG (Retrieval-Augmented Generation) Support

A Python library for processing text using OpenAI's GPT models with support for
context injection from external knowledge sources.

Requirements:
    pip install openai numpy sentence-transformers faiss-cpu tiktoken

Usage:
    from gpt_rag_processor import GPTTextProcessor
    
    processor = GPTTextProcessor(api_key="your-openai-api-key")
    
    # Add documents to the knowledge base
    documents = ["Document 1 content...", "Document 2 content..."]
    processor.add_documents(documents)
    
    # Process text with RAG context
    result = processor.process_with_rag("What is the main topic?")
    print(result)
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tiktoken
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a document in the knowledge base."""
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RAGResult:
    """Result from RAG-enhanced text processing."""
    response: str
    retrieved_documents: List[Document]
    confidence_scores: List[float]
    tokens_used: int


class EmbeddingManager:
    """Manages document embeddings and similarity search."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.documents: List[Document] = []
        
    def add_documents(self, documents: List[Union[str, Document]]) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document contents or Document objects
        """
        doc_objects = []
        texts = []
        
        for doc in documents:
            if isinstance(doc, str):
                doc_obj = Document(content=doc)
            else:
                doc_obj = doc
            
            doc_objects.append(doc_obj)
            texts.append(doc_obj.content)
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store documents with embeddings
        for doc_obj, embedding in zip(doc_objects, embeddings):
            doc_obj.embedding = embedding
            self.documents.append(doc_obj)
        
        logger.info(f"Added {len(doc_objects)} documents to knowledge base")
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (document, score) tuples
        """
        if len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save_index(self, filepath: str) -> None:
        """Save the FAISS index and documents to disk."""
        faiss.write_index(self.index, f"{filepath}.index")
        
        # Save documents metadata
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding.tolist() if doc.embedding is not None else None
            })
        
        with open(f"{filepath}.docs", "w", encoding="utf-8") as f:
            json.dump(docs_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved index to {filepath}")
    
    def load_index(self, filepath: str) -> None:
        """Load the FAISS index and documents from disk."""
        self.index = faiss.read_index(f"{filepath}.index")
        
        with open(f"{filepath}.docs", "r", encoding="utf-8") as f:
            docs_data = json.load(f)
        
        self.documents = []
        for doc_data in docs_data:
            doc = Document(
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                embedding=np.array(doc_data["embedding"]) if doc_data["embedding"] else None
            )
            self.documents.append(doc)
        
        logger.info(f"Loaded index from {filepath}")


class GPTTextProcessor:
    """Main class for processing text with OpenAI GPT and RAG support."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_context_tokens: int = 4000
    ):
        """
        Initialize the GPT text processor.
        
        Args:
            api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
            model: OpenAI model to use
            embedding_model: Sentence transformer model for embeddings
            max_context_tokens: Maximum tokens to use for context
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.max_context_tokens = max_context_tokens
        
        # Initialize embedding manager
        self.embedding_manager = EmbeddingManager(embedding_model)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized GPT processor with model: {model}")
    
    def add_documents(self, documents: List[Union[str, Document]]) -> None:
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document contents or Document objects
        """
        self.embedding_manager.add_documents(documents)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _build_context(self, query: str, retrieved_docs: List[tuple]) -> str:
        """
        Build context from retrieved documents.
        
        Args:
            query: Original query
            retrieved_docs: List of (document, score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = ["Retrieved context:"]
        current_tokens = self.count_tokens("Retrieved context:")
        
        for i, (doc, score) in enumerate(retrieved_docs):
            doc_text = f"\n[Document {i+1}] {doc.content}"
            doc_tokens = self.count_tokens(doc_text)
            
            # Check if adding this document would exceed token limit
            if current_tokens + doc_tokens > self.max_context_tokens:
                break
            
            context_parts.append(doc_text)
            current_tokens += doc_tokens
        
        return "\n".join(context_parts)
    
    def process_with_rag(
        self, 
        query: str, 
        top_k: int = 5,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> RAGResult:
        """
        Process text with RAG context injection.
        
        Args:
            query: Input query/text to process
            top_k: Number of documents to retrieve for context
            system_prompt: Custom system prompt
            temperature: OpenAI temperature parameter
            
        Returns:
            RAGResult object with response and metadata
        """
        # Retrieve relevant documents
        retrieved_docs = self.embedding_manager.search(query, top_k)
        
        if not retrieved_docs:
            logger.warning("No documents found in knowledge base")
            return self._process_without_rag(query, system_prompt, temperature)
        
        # Build context
        context = self._build_context(query, retrieved_docs)
        
        # Prepare messages
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system", 
                "content": "You are a helpful assistant. Use the provided context to answer questions accurately. If the context doesn't contain relevant information, say so."
            })
        
        # Add context and query
        full_prompt = f"{context}\n\nUser Query: {query}"
        messages.append({"role": "user", "content": full_prompt})
        
        # Make API call
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            result = RAGResult(
                response=response.choices[0].message.content,
                retrieved_documents=[doc for doc, _ in retrieved_docs],
                confidence_scores=[score for _, score in retrieved_docs],
                tokens_used=response.usage.total_tokens
            )
            
            logger.info(f"Processed query with {len(retrieved_docs)} context documents")
            return result
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def _process_without_rag(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> RAGResult:
        """Process text without RAG context."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": query})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            
            return RAGResult(
                response=response.choices[0].message.content,
                retrieved_documents=[],
                confidence_scores=[],
                tokens_used=response.usage.total_tokens
            )
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def process_text(
        self, 
        text: str, 
        instruction: str = "Process this text:",
        use_rag: bool = True,
        **kwargs
    ) -> RAGResult:
        """
        Process text with optional RAG enhancement.
        
        Args:
            text: Text to process
            instruction: Processing instruction
            use_rag: Whether to use RAG context
            **kwargs: Additional arguments for process_with_rag
            
        Returns:
            RAGResult object
        """
        full_query = f"{instruction}\n\n{text}"
        
        if use_rag:
            return self.process_with_rag(full_query, **kwargs)
        else:
            return self._process_without_rag(full_query, **kwargs)
    
    def save_knowledge_base(self, filepath: str) -> None:
        """Save the knowledge base to disk."""
        self.embedding_manager.save_index(filepath)
    
    def load_knowledge_base(self, filepath: str) -> None:
        """Load the knowledge base from disk."""
        self.embedding_manager.load_index(filepath)

def ask_llm(text: str) -> str:
    processor = GPTTextProcessor(
        api_key="<open AI key>",  # Replace with your API key
        model="gpt-3.5-turbo"
    )
    result = processor.process_text(text)
    return result.response


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = GPTTextProcessor(
        api_key="<open API key>",  # Replace with your API key
        model="gpt-3.5-turbo"
    )
    
    # Add sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
        "Natural language processing (NLP) is a field of AI that gives computers the ability to understand human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns in data.",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation for better responses."
    ]
    
    processor.add_documents(documents)
    
    # Process queries with RAG
    query = "What is machine learning and how does it relate to AI?"
    result = processor.process_with_rag(query)
    
    print(f"Query: {query}")
    print(f"Response: {result.response}")
    print(f"Retrieved {len(result.retrieved_documents)} documents")
    print(f"Tokens used: {result.tokens_used}")
    
    # Save knowledge base
    # processor.save_knowledge_base("my_knowledge_base")
    
    # Load knowledge base (example)
    # processor.load_knowledge_base("my_knowledge_base")
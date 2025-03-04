import os
import csv
import logging
from pathlib import Path
from typing import List, Optional, Union

from haystack import Document
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.converters import (
    HTMLToDocument,
    MarkdownToDocument,
    JSONConverter,
    PDFMinerToDocument,
    TextFileToDocument,
    DOCXToDocument,
)
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter

logger = logging.getLogger(__name__)

def get_supported_file_extensions() -> List[str]:
    """Return a list of supported file extensions for conversion."""
    return [".pdf", ".txt", ".docx", ".md", ".html", ".htm", ".json"]

def fetch_urls_from_csv(csv_path: Path) -> List[str]:
    """Extract URLs from a CSV file.
    
    Args:
        csv_path: Path to the CSV file containing URLs
        
    Returns:
        A list of URLs extracted from the CSV
    """
    if not csv_path.exists():
        logger.warning(f"URL CSV file not found at {csv_path}")
        return []
    
    urls = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) > 0 and row[0].strip():
                    urls.append(row[0].strip())
    except Exception as e:
        logger.error(f"Error reading URL CSV file: {e}")
    
    logger.info(f"Extracted {len(urls)} URLs from {csv_path}")
    return urls

def fetch_documents_from_urls(urls: List[str]) -> List[Document]:
    """Fetch documents from a list of URLs.
    
    Args:
        urls: List of URLs to fetch
        
    Returns:
        List of Document objects created from the fetched content
    """
    if not urls:
        return []
    
    documents = []
    web_fetcher = LinkContentFetcher()
    html_converter = HTMLToDocument()
    
    for url in urls:
        try:
            # Fetch content from URL
            fetch_results = web_fetcher.run({"urls": [url]})
            web_pages = fetch_results["web_pages"]
            
            if web_pages:
                # Convert HTML to Document objects
                converter_results = html_converter.run({"html_documents": web_pages})
                url_documents = converter_results["documents"]
                
                # Add source URL to metadata
                for doc in url_documents:
                    if not doc.meta:
                        doc.meta = {}
                    doc.meta["source"] = url
                    doc.meta["source_type"] = "url"
                
                documents.extend(url_documents)
                logger.info(f"Successfully fetched and converted URL: {url}")
            else:
                logger.warning(f"Failed to fetch content from URL: {url}")
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
    
    logger.info(f"Fetched {len(documents)} documents from {len(urls)} URLs")
    return documents

def get_file_converter(file_path: Path):
    """Return the appropriate converter for a given file path."""
    file_ext = file_path.suffix.lower()
    
    if file_ext == ".pdf":
        return PDFMinerToDocument()
    elif file_ext == ".txt":
        return TextFileToDocument()
    elif file_ext == ".docx":
        return DOCXToDocument()
    elif file_ext == ".md":
        return MarkdownToDocument()
    elif file_ext in [".html", ".htm"]:
        return HTMLToDocument()
    elif file_ext == ".json":
        return JSONConverter(jq_schema=".documents[]", content_key="content")
    else:
        return None

def convert_local_files(corpus_dir: Path) -> List[Document]:
    """Convert all supported files in a directory to Document objects.
    
    Args:
        corpus_dir: Path to directory containing files to convert
        
    Returns:
        List of Document objects created from local files
    """
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        logger.warning(f"Corpus directory not found or not a directory: {corpus_dir}")
        return []
    
    documents = []
    supported_extensions = get_supported_file_extensions()
    
    # Find all files with supported extensions
    for file_path in corpus_dir.glob("**/*"):
        print(f"Processing file: {file_path}")
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                # Skip the urls.csv file as it's handled separately
                if file_path.name == "urls.csv":
                    continue
                
                converter = get_file_converter(file_path)
                if converter:
                    # Convert file to Document objects
                    converter_results = converter.run(sources= [str(file_path)])
                    file_documents = converter_results["documents"]
                    
                    # Add source path to metadata
                    for doc in file_documents:
                        if not doc.meta:
                            doc.meta = {}
                        doc.meta["source"] = str(file_path)
                        doc.meta["source_type"] = "file"
                        doc.meta["file_type"] = file_path.suffix.lower()
                    
                    documents.extend(file_documents)
                    logger.info(f"Successfully converted file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
    
    print(f"Converted {len(documents)} documents from local files")
    return documents

def preprocess_documents(documents: List[Document], 
                         clean: bool = True, 
                         split: bool = True,
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200) -> List[Document]:
    """Clean and split documents if requested.
    
    Args:
        documents: List of documents to preprocess
        clean: Whether to clean the documents
        split: Whether to split the documents
        chunk_size: Size of document chunks if splitting
        chunk_overlap: Overlap between chunks if splitting
        
    Returns:
        List of preprocessed Document objects
    """
    if not documents:
        return []
    
    processed_docs = documents
    
    # Clean documents if requested
    if clean:
        cleaner = DocumentCleaner()
        cleaner_results = cleaner.run(processed_docs)
        processed_docs = cleaner_results["documents"]
    
    # Split documents if requested
    if split:
        splitter = DocumentSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap
        )
        splitter_results = splitter.run(documents=processed_docs)
        processed_docs = splitter_results["documents"]
    
    return processed_docs

def get_all_documents(corpus_dir: Union[str, Path], 
                      clean: bool = True,
                      split: bool = True,
                      chunk_size: int = 1000,
                      chunk_overlap: int = 200) -> List[Document]:
    """Process all documents in a corpus directory and return a unified list.
    
    Args:
        corpus_dir: Path to the corpus directory
        clean: Whether to clean the documents
        split: Whether to split the documents
        chunk_size: Size of document chunks if splitting
        chunk_overlap: Overlap between chunks if splitting
        
    Returns:
        List of Document objects from all sources
    """
    corpus_path = Path(corpus_dir)
    all_documents = []
    
    # 1. Convert local files
    file_documents = convert_local_files(corpus_path)
    all_documents.extend(file_documents)
    
    # 2. Fetch and convert URLs if urls.csv exists
    urls_csv_path = corpus_path / "urls.csv"
    if urls_csv_path.exists():
        urls = fetch_urls_from_csv(urls_csv_path)
        url_documents = fetch_documents_from_urls(urls)
        all_documents.extend(url_documents)
    
    # 3. Preprocess all documents
    processed_documents = preprocess_documents(
        all_documents,
        clean=clean,
        split=split,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    print(f"Total documents processed: {len(processed_documents)}")
    return processed_documents 
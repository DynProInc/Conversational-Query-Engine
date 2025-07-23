"""
Text processing utilities for Conversational Query Engine.
This module provides functions for processing, chunking, and analyzing text data.
"""

from typing import List, Dict, Any, Union, Optional, Tuple
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import logging

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize logger
logger = logging.getLogger(__name__)

# Load spaCy model - using the small English model for efficiency
try:
    nlp = spacy.load("en_core_web_sm")
    # Ensure we have the sentencizer component
    if 'sentencizer' not in nlp.pipe_names:
        sentencizer = nlp.add_pipe('sentencizer')
        logger.info("Added sentencizer component to spaCy pipeline")
except OSError:
    logger.warning("Spacy model 'en_core_web_sm' not found. Using default language class instead.")
    nlp = spacy.blank("en")
    # Add sentencizer to blank model
    if 'sentencizer' not in nlp.pipe_names:
        sentencizer = nlp.add_pipe('sentencizer')
        logger.info("Added sentencizer component to default spaCy pipeline")

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Clean text first
    text = clean_text(text)
    
    # Split into sentences
    return sent_tokenize(text)

def split_text_by_tokens(text: str, max_tokens: int = 512, overlap: int = 100) -> List[str]:
    """
    Split text into chunks based on token count with overlap.
    
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk
        overlap: Number of tokens of overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Split into chunks
    chunks = []
    start_idx = 0
    
    while start_idx < len(tokens):
        # Calculate end index for this chunk
        end_idx = min(start_idx + max_tokens, len(tokens))
        
        # Extract tokens for this chunk
        chunk_tokens = tokens[start_idx:end_idx]
        
        # Join tokens back into text
        chunk_text = ' '.join(chunk_tokens)
        
        # Add chunk to list
        chunks.append(chunk_text)
        
        # Move start index for next chunk, considering overlap
        start_idx += max_tokens - overlap
        
        # Ensure we don't get stuck in a loop if overlap >= max_tokens
        if start_idx <= 0:
            start_idx = end_idx
    
    return chunks

def split_text_by_semantic_units(text: str, max_tokens: int = 512) -> List[str]:
    """
    Split text into semantically coherent chunks using spaCy.
    
    Args:
        text: Input text to split
        max_tokens: Maximum number of tokens per chunk
        
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Process text with spaCy
    doc = nlp(text)
    
    # Split by paragraphs and then by sentences
    chunks = []
    current_chunk = []
    current_token_count = 0
    
    # Use spaCy's sentence segmentation
    for sent in doc.sents:
        sent_text = sent.text.strip()
        sent_tokens = len(sent_text.split())
        
        # If a single sentence is longer than max_tokens, use token-based chunking
        if sent_tokens > max_tokens:
            sent_chunks = split_text_by_tokens(sent_text, max_tokens)
            chunks.extend(sent_chunks)
            continue
        
        # If adding this sentence would exceed max_tokens, start a new chunk
        if current_token_count + sent_tokens > max_tokens and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_token_count = 0
        
        # Add sentence to current chunk
        current_chunk.append(sent_text)
        current_token_count += sent_tokens
    
    # Don't forget to add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def chunk_schema_definition(schema_text: str, max_tokens: int = 1024) -> List[Dict[str, Any]]:
    """
    Chunk database schema definitions into semantically meaningful units.
    
    Args:
        schema_text: Text containing schema definitions
        max_tokens: Maximum number of tokens per chunk
        
    Returns:
        List of dictionaries containing chunk text and metadata
    """
    # Split by table definitions (assuming schema text has a specific format)
    table_pattern = r'(CREATE\s+TABLE\s+[^;]+;)'
    table_chunks = re.findall(table_pattern, schema_text, re.IGNORECASE | re.DOTALL)
    
    result_chunks = []
    
    for table_chunk in table_chunks:
        # Extract table name
        table_name_match = re.search(r'CREATE\s+TABLE\s+([^\s(]+)', table_chunk, re.IGNORECASE)
        table_name = table_name_match.group(1) if table_name_match else "Unknown"
        
        # Clean table name by removing quotes, brackets, etc.
        table_name = re.sub(r'["\[\]`]', '', table_name)
        
        # If the table definition is too long, split it
        if len(table_chunk.split()) > max_tokens:
            # Split by columns
            column_chunks = []
            
            # Extract the part between parentheses
            columns_match = re.search(r'\((.*)\)', table_chunk, re.DOTALL)
            
            if columns_match:
                columns_text = columns_match.group(1)
                
                # Split columns by comma (but not inside parentheses)
                depth = 0
                current_column = ""
                
                for char in columns_text:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                    
                    current_column += char
                    
                    if char == ',' and depth == 0:
                        column_chunks.append(current_column.strip())
                        current_column = ""
                
                # Add the last column
                if current_column.strip():
                    column_chunks.append(current_column.strip())
                
                # Group columns into chunks
                current_chunk = f"CREATE TABLE {table_name} ("
                current_tokens = len(current_chunk.split())
                
                for i, column in enumerate(column_chunks):
                    column_tokens = len(column.split())
                    
                    if current_tokens + column_tokens > max_tokens:
                        # Finalize current chunk and start a new one
                        if i > 0:  # If not the first column, need to replace the last comma
                            current_chunk = current_chunk[:-1] + ");"
                        else:
                            current_chunk += ");"
                        
                        result_chunks.append({
                            "text": current_chunk,
                            "metadata": {
                                "content_type": "schema",
                                "table_name": table_name,
                                "is_complete": False
                            }
                        })
                        
                        # Start a new chunk
                        current_chunk = f"CREATE TABLE {table_name} ("
                        current_tokens = len(current_chunk.split())
                    
                    # Add column to current chunk
                    current_chunk += column + ", "
                    current_tokens += column_tokens
                
                # Finalize the last chunk
                if current_chunk.endswith(", "):
                    current_chunk = current_chunk[:-2] + ");"
                
                result_chunks.append({
                    "text": current_chunk,
                    "metadata": {
                        "content_type": "schema",
                        "table_name": table_name,
                        "is_complete": True
                    }
                })
            else:
                # If we can't extract columns properly, just add the whole table as one chunk
                result_chunks.append({
                    "text": table_chunk,
                    "metadata": {
                        "content_type": "schema",
                        "table_name": table_name,
                        "is_complete": True
                    }
                })
        else:
            # If table definition is small enough, add it as one chunk
            result_chunks.append({
                "text": table_chunk,
                "metadata": {
                    "content_type": "schema",
                    "table_name": table_name,
                    "is_complete": True
                }
            })
    
    return result_chunks

def extract_table_and_column_info(schema_text: str) -> List[Dict[str, Any]]:
    """
    Extract table and column information from schema text.
    
    Args:
        schema_text: Text containing schema definitions
        
    Returns:
        List of dictionaries containing table/column information
    """
    # Split by table definitions
    table_pattern = r'CREATE\s+TABLE\s+([^\s(]+)\s*\((.*?)\);'
    tables = re.findall(table_pattern, schema_text, re.IGNORECASE | re.DOTALL)
    
    result = []
    
    for table_name, columns_text in tables:
        # Clean table name
        table_name = re.sub(r'["\[\]`]', '', table_name)
        
        # Split columns
        column_entries = []
        
        # Split by commas (but not inside parentheses)
        depth = 0
        current_column = ""
        
        for char in columns_text:
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            current_column += char
            
            if char == ',' and depth == 0:
                column_entries.append(current_column.strip())
                current_column = ""
        
        # Add the last column
        if current_column.strip():
            column_entries.append(current_column.strip())
        
        # Extract column information
        columns = []
        for entry in column_entries:
            # Skip if this is not actually a column definition (like constraints)
            if not re.match(r'^\s*[A-Za-z0-9_"\[\]`]+\s', entry):
                continue
            
            # Extract column name and type
            column_match = re.match(r'^\s*([A-Za-z0-9_"\[\]`]+)\s+([A-Za-z0-9_()]+)', entry)
            
            if column_match:
                col_name = re.sub(r'["\[\]`]', '', column_match.group(1))
                col_type = column_match.group(2)
                
                # Extract additional attributes
                is_nullable = "NOT NULL" not in entry.upper()
                is_primary = "PRIMARY KEY" in entry.upper()
                
                column_info = {
                    "name": col_name,
                    "type": col_type,
                    "nullable": is_nullable,
                    "primary_key": is_primary
                }
                
                # Extract default value if present
                default_match = re.search(r'DEFAULT\s+([^,)]+)', entry, re.IGNORECASE)
                if default_match:
                    column_info["default"] = default_match.group(1).strip()
                
                columns.append(column_info)
        
        # Add table info to result
        result.append({
            "table_name": table_name,
            "columns": columns
        })
    
    return result

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract key terms from a text.
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        List of extracted keywords
    """
    if not text:
        return []
    
    # Process with spaCy
    doc = nlp(text)
    
    # Extract nouns, proper nouns, and noun chunks as potential keywords
    keywords = []
    
    # Add named entities
    for ent in doc.ents:
        keywords.append(ent.text.lower())
    
    # Add noun chunks
    for chunk in doc.noun_chunks:
        keywords.append(chunk.text.lower())
    
    # Add important individual tokens (nouns, proper nouns, etc.)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and token.is_alpha:
            keywords.append(token.text.lower())
    
    # Remove duplicates and sort by frequency
    keyword_freq = {}
    for keyword in keywords:
        keyword = keyword.strip()
        if keyword and len(keyword) > 1:  # Ignore single-character keywords
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    # Sort by frequency
    sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N keywords
    return [k for k, _ in sorted_keywords[:max_keywords]]

def preprocess_user_query(query: str) -> Tuple[str, List[str]]:
    """
    Preprocess user query for improved retrieval.
    
    Args:
        query: User's natural language query
        
    Returns:
        Tuple of (cleaned query, extracted keywords)
    """
    # Clean the query
    cleaned_query = clean_text(query)
    
    # Extract keywords for search optimization
    keywords = extract_keywords(cleaned_query)
    
    return cleaned_query, keywords

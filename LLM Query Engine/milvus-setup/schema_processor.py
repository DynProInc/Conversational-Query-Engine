#!/usr/bin/env python3
"""
Schema Processor for Multi-Client RAG
=====================================

Processes database schema information from various client formats
with enhanced handling of combined DATABASE.SCHEMA values.

Author: DynProInc
Date: 2025-07-22
"""

import os
import logging
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any, Union
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SchemaProcessor")

# Default column mapping configuration
DEFAULT_COLUMN_MAPPING = {
    'db_schema': ['DB_SCHEMA', 'SCHEMA', 'DATABASE_SCHEMA'],
    'table_name': ['TABLE_NAME', 'TABLENAME', 'TABLE', 'VIEW_NAME', 'VIEWNAME', 'VIEW', 'OBJECT_NAME'],
    'column_name': ['COLUMN_NAME', 'COLUMNNAME', 'COLUMN', 'FIELD_NAME', 'FIELD'],
    'data_type': ['DATA_TYPE', 'DATATYPE', 'TYPE', 'COLUMN_TYPE'],
    'description': ['DESCRIPTION', 'DESC', 'COLUMN_DESCRIPTION', 'COMMENT'],
    'distinct_values': ['DISTINCT_VALUES', 'DISTINCT_VALS', 'VALUES', 'SAMPLE_VALUES']
}

@dataclass
class SchemaRecord:
    """Data class for database schema records with enhanced DB_SCHEMA handling"""
    client_id: str
    db_schema: str
    table_name: str
    column_name: str
    data_type: str = ''
    description: str = ''
    distinct_values: Optional[str] = None
    combined_text: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Generate combined text for embedding if not provided"""
        if not self.combined_text:
            self.combined_text = self.generate_combined_text()
    
    def generate_combined_text(self) -> str:
        """
        Generate combined text representation for embedding
        Handles the DB_SCHEMA format with DATABASE.SCHEMA combined values
        """
        # Extract database and schema if in combined format
        database, schema = self.extract_db_schema()
        
        # Skip empty values to avoid misleading the model
        components = []
        
        if database:
            components.append(f"Database: {database}")
        if schema:
            components.append(f"Schema: {schema}")
        if self.table_name:
            components.append(f"Table: {self.table_name}")
        if self.column_name:
            components.append(f"Column: {self.column_name}")
        if self.data_type:
            components.append(f"Data Type: {self.data_type}")
        if self.description:
            components.append(f"Description: {self.description}")
        
        # Only add distinct values if they exist and aren't empty
        if self.distinct_values and self.distinct_values != "N/A" and self.distinct_values.strip():
            components.append(f"Distinct Values: {self.distinct_values}")
        
        # Join all components with newlines
        return "\n".join(components)
    
    def extract_db_schema(self) -> Tuple[str, str]:
        """
        Extract database and schema from DB_SCHEMA field
        Handles NaN and non-string values safely
        
        Returns:
            Tuple of (database, schema)
        """
        # Handle NaN, None, or non-string values
        if not isinstance(self.db_schema, str):
            return "", ""
            
        if not self.db_schema or "." not in self.db_schema:
            return "", self.db_schema or ""
        
        parts = self.db_schema.split(".", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        
        return "", self.db_schema


class SchemaProcessor:
    """Processes schema data dictionaries for multiple clients"""
    
    def __init__(self, column_mapping: Dict[str, List[str]] = None):
        """Initialize schema processor with optional custom column mapping
        
        Args:
            column_mapping: Custom column name mappings. If None, uses default mapping.
        """
        self.logger = logging.getLogger("SchemaProcessor")
        self.column_mapping = column_mapping or DEFAULT_COLUMN_MAPPING.copy()
        
        # Client-specific column mappings
        self.client_mappings = {}
        
        # Try to load client-specific mappings from config
        self._load_client_mappings()
        
    def _load_client_mappings(self):
        """Load client-specific column mappings from config file"""
        # Get the path to config directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, 'config', 'column_mappings.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    self.client_mappings = json.load(f)
                
                # Count actual clients (excluding _default entry)
                actual_clients = [client for client in self.client_mappings.keys() if client != '_default']
                self.logger.info(f"Loaded column mappings for {len(actual_clients)} clients")
            except Exception as e:
                self.logger.warning(f"Failed to load client column mappings: {e}")
    
    def _get_column_name(self, client_id: str, field: str, df_columns: List[str]) -> Optional[str]:
        """
        Get the actual column name from the dataframe based on mappings
        
        Args:
            client_id: Client identifier
            field: Field name in SchemaRecord
            df_columns: Available columns in dataframe
            
        Returns:
            Actual column name if found, None otherwise
        """
        # Priority order for mappings:
        # 1. Client-specific mapping
        # 2. _default mapping from client_mappings.json
        # 3. DEFAULT_COLUMN_MAPPING from code
        
        # Check for client-specific mapping first
        client_mapping = self.client_mappings.get(client_id, {})
        default_mapping = self.client_mappings.get('_default', {})
        
        # Get possible column names with proper fallback order
        possible_names = []
        
        # 1. Try client-specific mapping
        if field in client_mapping:
            possible_names.extend(client_mapping[field])
            
        # 2. Try _default mapping from configuration
        if field in default_mapping:
            for name in default_mapping[field]:
                if name not in possible_names:  # Avoid duplicates
                    possible_names.append(name)
        
        # 3. Try built-in default mapping
        if field in self.column_mapping:
            for name in self.column_mapping[field]:
                if name not in possible_names:  # Avoid duplicates
                    possible_names.append(name)
        
        # Try each possible name in the prioritized order
        for name in possible_names:
            if name in df_columns:
                return name
        
        # If required field, log warning
        if field in ['table_name', 'column_name']:
            self.logger.warning(f"Required field '{field}' not found for client {client_id}")
            self.logger.debug(f"Available columns: {df_columns}")
            self.logger.debug(f"Tried column names: {possible_names}")
        
        return None
        
    def _safe_get_value(self, row, column):
        """
        Safely extract a value from a dataframe row
        Handles missing columns and NaN values
        
        Args:
            row: DataFrame row
            column: Column name to extract
            
        Returns:
            String value or empty string if not available
        """
        # Check if column exists
        if column is None or column not in row:
            return ""
        
        # Get value and handle NaN, None
        value = row[column]
        if pd.isna(value) or value is None:
            return ""
        
        # Convert to string if it's not already
        if not isinstance(value, str):
            return str(value)
            
        return value
    
    def process_csv_data(self, client_id: str, csv_data) -> List[SchemaRecord]:
        """
        Process CSV data into schema records
        
        Args:
            client_id: Client identifier
            csv_data: Either a pandas DataFrame or path to CSV file
            
        Returns:
            List of SchemaRecord objects
        """
        import pandas as pd
        
        # Handle case where a file path is passed instead of DataFrame
        if isinstance(csv_data, str):
            # Try multiple encodings in order of likelihood
            encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
            df_data = None
            
            for encoding in encodings:
                try:
                    self.logger.info(f"Loading CSV from path: {csv_data} with encoding: {encoding}")
                    df_data = pd.read_csv(csv_data, encoding=encoding)
                    self.logger.info(f"Successfully loaded CSV with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    self.logger.warning(f"Failed to load CSV with encoding: {encoding}, trying next encoding")
                except Exception as e:
                    self.logger.error(f"Failed to load CSV file {csv_data} with encoding {encoding}: {e}")
                    continue
            
            if df_data is None:
                self.logger.error(f"Failed to load CSV file {csv_data} with any encoding")
                return []
        else:
            df_data = csv_data
        
        records = []
        
        # Extract column names and check for required columns
        df_columns = list(df_data.columns)
        self.logger.debug(f"Found columns: {df_columns}")
        
        # Get mapped column names for this client
        col_db_schema = self._get_column_name(client_id, 'db_schema', df_columns)
        col_table_name = self._get_column_name(client_id, 'table_name', df_columns)
        col_column_name = self._get_column_name(client_id, 'column_name', df_columns)
        col_data_type = self._get_column_name(client_id, 'data_type', df_columns)
        col_description = self._get_column_name(client_id, 'description', df_columns)
        col_distinct_values = self._get_column_name(client_id, 'distinct_values', df_columns)
        
        # Log found column mappings
        self.logger.info(f"Using column mappings for {client_id}: " + 
                     f"db_schema={col_db_schema}, " + 
                     f"table_name={col_table_name}, " + 
                     f"column_name={col_column_name}, " + 
                     f"data_type={col_data_type}, " + 
                     f"description={col_description}, " + 
                     f"distinct_values={col_distinct_values}")
        
        # Exit early if required columns are not found
        if not col_table_name or not col_column_name:
            self.logger.error(f"Required columns not found for client {client_id}")
            return []
        
        for _, row in df_data.iterrows():
            db_schema = self._safe_get_value(row, col_db_schema) if col_db_schema else ''
            table_name = self._safe_get_value(row, col_table_name) if col_table_name else ''
            column_name = self._safe_get_value(row, col_column_name) if col_column_name else ''
            data_type = self._safe_get_value(row, col_data_type) if col_data_type else ''
            description = self._safe_get_value(row, col_description) if col_description else ''
            distinct_values = self._safe_get_value(row, col_distinct_values) if col_distinct_values else ''
            
            # Skip records without table or column name
            if not table_name or not column_name:
                continue
            
            # Create metadata from any extra columns
            metadata = {}
            for col in df_columns:
                # Skip None and mapped columns
                mapped_cols = [c for c in [col_db_schema, col_table_name, col_column_name, 
                              col_data_type, col_description, col_distinct_values] if c is not None]
                if col not in mapped_cols:
                    metadata[col] = self._safe_get_value(row, col)
                
            # Create schema record
            record = SchemaRecord(
                client_id=client_id,
                db_schema=db_schema,
                table_name=table_name,
                column_name=column_name,
                data_type=data_type,
                description=description,
                distinct_values=distinct_values,
                metadata=metadata
            )
            
            records.append(record)
        
        self.logger.info(f"Processed {len(records)} schema records for client {client_id}")
        return records
    
    def extract_metadata(self, record: SchemaRecord) -> Dict[str, str]:
        """
        Extract metadata from schema record
        
        Args:
            record: Schema record
            
        Returns:
            Dictionary with metadata
        """
        database, schema = record.extract_db_schema()
        
        return {
            "client_id": record.client_id,
            "database": database,
            "schema": schema,
            "db_schema": record.db_schema,
            "table_name": record.table_name,
            "column_name": record.column_name,
            "data_type": record.data_type
        }


# Helper functions for integration
def process_schema_dict(client_id: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process schema dictionary with enhanced DB_SCHEMA handling
    
    Args:
        client_id: Client identifier
        schema_dict: Dictionary with schema data
        
    Returns:
        Processed dictionary with embedding text and metadata
    """
    # Create schema record
    record = SchemaRecord(
        client_id=client_id,
        db_schema=schema_dict.get('db_schema', ""),
        table_name=schema_dict.get('table_name', ""),
        column_name=schema_dict.get('column_name', ""),
        data_type=schema_dict.get('data_type', ""),
        description=schema_dict.get('description', ""),
        distinct_values=schema_dict.get('distinct_values', "")
    )
    
    # Extract database and schema
    database, schema = record.extract_db_schema()
    
    # Return processed dictionary
    return {
        'text': record.combined_text,
        'metadata': {
            'client_id': client_id,
            'database': database,
            'schema': schema,
            'db_schema': record.db_schema,
            'table_name': record.table_name,
            'column_name': record.column_name,
            'data_type': record.data_type
        }
    }

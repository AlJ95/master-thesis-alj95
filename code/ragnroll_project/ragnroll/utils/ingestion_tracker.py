"""
Ingestion tracking system for RAGnRoll.

This module provides functionality to track ingested datasets and avoid redundant
ingestion operations by maintaining a CSV-based registry of processed corpora.
"""

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from datetime import datetime

import pandas as pd


@dataclass
class IngestionRecord:
    """Represents a single ingestion record."""
    corpus_path: str
    processing_config_hash: str
    index_id: str
    document_ids: List[str]
    timestamp: str
    status: str  # 'completed', 'partial', 'failed'


class IngestionTracker:
    """Manages tracking of ingested datasets to prevent redundant operations."""

    def __init__(self, tracking_file: Optional[str] = None):
        """
        Initialize the ingestion tracker.

        Args:
            tracking_file: Path to the CSV tracking file. If None, uses default location.
        """
        if tracking_file is None:
            # Default location relative to project root
            project_root = Path(__file__).parent.parent.parent
            self.tracking_file = project_root / "data" / "ingestion_tracking.csv"
        else:
            self.tracking_file = Path(tracking_file)

        # Ensure directory exists
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize CSV if it doesn't exist
        if not self.tracking_file.exists():
            self._initialize_tracking_file()

    def _initialize_tracking_file(self):
        """Create the tracking CSV file with headers."""
        with open(self.tracking_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'corpus_path',
                'processing_config_hash',
                'index_id',
                'document_ids',
                'timestamp',
                'status'
            ])

    def generate_index_id(self, corpus_path: str, processing_config: Dict[str, Any]) -> str:
        """
        Generate a deterministic index ID based on corpus path and processing configuration.

        Args:
            corpus_path: Path to the corpus directory
            processing_config: Dictionary containing processing parameters

        Returns:
            Deterministic index ID as hex string
        """
        # Create a normalized representation of the inputs
        index_data = {
            'corpus_path': str(Path(corpus_path).resolve()),
            'processing_config': self._normalize_config(processing_config)
        }

        # Generate hash
        index_str = json.dumps(index_data, sort_keys=True)
        return hashlib.sha256(index_str.encode('utf-8')).hexdigest()

    def generate_document_id(self, document_content: str, processing_config: Dict[str, Any]) -> str:
        """
        Generate a deterministic document ID based on content and processing config.

        Args:
            document_content: The document content
            processing_config: Processing configuration used

        Returns:
            Deterministic document ID as hex string
        """
        doc_data = {
            'content': document_content,
            'processing_config': self._normalize_config(processing_config)
        }

        doc_str = json.dumps(doc_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(doc_str.encode('utf-8')).hexdigest()

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration for consistent hashing."""
        # Sort keys and convert to strings for consistency
        normalized = {}
        for key in sorted(config.keys()):
            value = config[key]
            if isinstance(value, (list, tuple)):
                # Sort lists for consistency
                normalized[key] = sorted([str(v) for v in value])
            else:
                normalized[key] = str(value)
        return normalized

    def get_existing_record(self, index_id: str) -> Optional[IngestionRecord]:
        """
        Check if an index ID already exists in the tracking file.

        Args:
            index_id: The index ID to check

        Returns:
            IngestionRecord if found, None otherwise
        """
        if not self.tracking_file.exists():
            return None

        try:
            df = pd.read_csv(self.tracking_file)
            matching_rows = df[df['index_id'] == index_id]

            if len(matching_rows) == 0:
                return None

            # Return the most recent record
            row = matching_rows.iloc[-1]
            doc_ids_str = str(row['document_ids']) if pd.notna(row['document_ids']) else ""
            document_ids = doc_ids_str.split(',') if doc_ids_str else []
            return IngestionRecord(
                corpus_path=str(row['corpus_path']),
                processing_config_hash=str(row['processing_config_hash']),
                index_id=str(row['index_id']),
                document_ids=document_ids,
                timestamp=str(row['timestamp']),
                status=str(row['status'])
            )
        except Exception as e:
            print(f"Warning: Error reading tracking file: {e}")
            return None

    def record_ingestion(self, record: IngestionRecord):
        """
        Record a completed ingestion operation.

        Args:
            record: The ingestion record to save
        """
        with open(self.tracking_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.corpus_path,
                record.processing_config_hash,
                record.index_id,
                ','.join(record.document_ids),
                record.timestamp,
                record.status
            ])

    def get_missing_documents(self, index_id: str, required_document_ids: Set[str]) -> Set[str]:
        """
        Get the set of document IDs that are missing from an existing index.

        Args:
            index_id: The index ID to check
            required_document_ids: Set of document IDs that should be in the index

        Returns:
            Set of document IDs that are missing
        """
        existing_record = self.get_existing_record(index_id)
        if existing_record is None:
            return required_document_ids

        existing_doc_ids = set(existing_record.document_ids)
        return required_document_ids - existing_doc_ids

    def update_record_status(self, index_id: str, status: str, new_document_ids: Optional[List[str]] = None):
        """
        Update the status of an existing record.

        Args:
            index_id: The index ID to update
            status: New status ('completed', 'partial', 'failed')
            new_document_ids: Additional document IDs to add (optional)
        """
        if not self.tracking_file.exists():
            return

        try:
            df = pd.read_csv(self.tracking_file)

            # Find the record
            mask = df['index_id'] == index_id
            if not mask.any():
                return

            # Update status
            df.loc[mask, 'status'] = status
            df.loc[mask, 'timestamp'] = datetime.now().isoformat()

            # Add new document IDs if provided
            if new_document_ids:
                for idx in df[mask].index:
                    existing_ids = df.at[idx, 'document_ids']
                    if pd.isna(existing_ids):
                        existing_ids = ''
                    existing_ids_str = str(existing_ids) if existing_ids else ''
                    existing_set = set(existing_ids_str.split(',')) if existing_ids_str else set()
                    existing_set.update(new_document_ids)
                    df.at[idx, 'document_ids'] = ','.join(sorted(existing_set))

            # Save back to file
            df.to_csv(self.tracking_file, index=False)

        except Exception as e:
            print(f"Warning: Error updating tracking file: {e}")

    def list_ingested_corpora(self) -> pd.DataFrame:
        """
        Get a DataFrame of all ingested corpora.

        Returns:
            DataFrame with ingestion records
        """
        if not self.tracking_file.exists():
            return pd.DataFrame()

        try:
            return pd.read_csv(self.tracking_file)
        except Exception as e:
            print(f"Warning: Error reading tracking file: {e}")
            return pd.DataFrame()


# Global instance for easy access
tracker = IngestionTracker()

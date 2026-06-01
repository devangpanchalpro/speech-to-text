"""
Database Manager for CoreInventory Audio Pipeline.

Handles PostgreSQL connection management, table initialization,
and CRUD operations for consultation_records.
"""

import os
import json
import psycopg2
from psycopg2.extras import Json, RealDictCursor
from datetime import datetime
from typing import Dict, Optional, List


class DatabaseManager:
    """Manages PostgreSQL database connections and operations."""

    # SQL for creating the consultation_records table
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS consultation_records (
        id SERIAL PRIMARY KEY,
        original_file VARCHAR(255) NOT NULL,
        processed_file VARCHAR(255) NOT NULL,
        detected_language VARCHAR(50),
        doctor_name VARCHAR(255),
        patient_name VARCHAR(255),
        transcript TEXT NOT NULL,
        transcript_english TEXT NOT NULL,
        identification JSONB NOT NULL DEFAULT '{}',
        hmis_data JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self):
        """Initialize database manager with connection parameters from environment."""
        self.connection_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "dbname": os.getenv("DB_NAME", "voice_to_rx"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
        }
        self._conn = None

    def _get_connection(self):
        """Get or create a database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(**self.connection_params)
        return self._conn

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def initialize_db(self):
        """Create the consultation_records table if it doesn't exist."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_TABLE_SQL)
            conn.commit()
            print("✅ Database initialized: 'consultation_records' table is ready.")
        except Exception as e:
            conn.rollback()
            print(f"❌ Database initialization error: {e}")
            raise

    def insert_record(self, result: Dict) -> Optional[int]:
        """
        Insert a pipeline result into the consultation_records table.

        Args:
            result: The full pipeline result dictionary containing:
                - metadata (original_file, processed_file, detected_language)
                - transcript, transcript_english
                - identification (doctor/patient info)
                - hmis (HMIS-formatted data)

        Returns:
            The ID of the inserted record, or None on failure.
        """
        conn = self._get_connection()

        metadata = result.get("metadata", {})
        identification = result.get("identification", {})
        hmis_data = result.get("hmis", {})

        try:
            with conn.cursor() as cur:
                # Find the lowest available ID (filling any gaps from deletions, starting from 1)
                cur.execute(
                    """
                    SELECT COALESCE(
                        (
                            SELECT MIN(gap.id)
                            FROM (
                                SELECT 1 AS id
                                UNION ALL
                                SELECT id + 1 FROM consultation_records
                            ) gap
                            WHERE gap.id NOT IN (SELECT id FROM consultation_records)
                        ),
                        1
                    );
                    """
                )
                next_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO consultation_records 
                        (id, original_file, processed_file, detected_language,
                         doctor_name, patient_name,
                         transcript, transcript_english,
                         identification, hmis_data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """,
                    (
                        next_id,
                        metadata.get("original_file", ""),
                        metadata.get("processed_file", ""),
                        metadata.get("detected_language", ""),
                        identification.get("doctor", {}).get("name", "Unknown"),
                        identification.get("patient", {}).get("name", "Unknown"),
                        result.get("transcript", ""),
                        result.get("transcript_english", ""),
                        Json(identification),
                        Json(hmis_data),
                    ),
                )
                record_id = cur.fetchone()[0]
            conn.commit()
            print(f"✅ Record saved to database with ID: {record_id}")
            return record_id

        except Exception as e:
            conn.rollback()
            print(f"❌ Database insert error: {e}")
            return None

    def get_all_records(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """
        Retrieve all consultation records, ordered by most recent first.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            A list of record dictionaries.
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, original_file, processed_file, detected_language,
                           doctor_name, patient_name,
                           transcript, transcript_english,
                           identification, hmis_data, created_at
                    FROM consultation_records
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s;
                    """,
                    (limit, offset),
                )
                rows = cur.fetchall()

            # Convert rows to serializable dicts
            records = []
            for row in rows:
                record = dict(row)
                # Convert datetime to ISO string
                if isinstance(record.get("created_at"), datetime):
                    record["created_at"] = record["created_at"].isoformat()
                records.append(record)

            return records

        except Exception as e:
            print(f"❌ Database query error: {e}")
            return []

    def get_record_by_id(self, record_id: int) -> Optional[Dict]:
        """
        Retrieve a single consultation record by ID.

        Args:
            record_id: The ID of the record to retrieve.

        Returns:
            A record dictionary, or None if not found.
        """
        conn = self._get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, original_file, processed_file, detected_language,
                           doctor_name, patient_name,
                           transcript, transcript_english,
                           identification, hmis_data, created_at
                    FROM consultation_records
                    WHERE id = %s;
                    """,
                    (record_id,),
                )
                row = cur.fetchone()

            if row:
                record = dict(row)
                if isinstance(record.get("created_at"), datetime):
                    record["created_at"] = record["created_at"].isoformat()
                return record
            return None

        except Exception as e:
            print(f"❌ Database query error: {e}")
            return None

    def get_record_count(self) -> int:
        """Return the total number of consultation records."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM consultation_records;")
                return cur.fetchone()[0]
        except Exception as e:
            print(f"❌ Database count error: {e}")
            return 0

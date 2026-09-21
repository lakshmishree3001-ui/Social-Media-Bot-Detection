"""
backend/services/pipeline/__init__.py
Ingestion pipeline package. Exposes the top-level run_ingestion_pipeline entry point.
"""
from backend.services.pipeline.runner import run_ingestion_pipeline

__all__ = ["run_ingestion_pipeline"]

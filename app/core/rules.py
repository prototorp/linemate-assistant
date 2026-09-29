"""Business rules shared by the API layer and the RAG layer
"""

from datetime import date

from app.core.config import STALE_THRESHOLD_DAYS
from app.models.enums import DocumentCategory


def is_review_exempt(category: DocumentCategory | str) -> bool:
    """Incident Reports are point-in-time records, not maintained guidance, so they never go stale"""
    return DocumentCategory(category) == DocumentCategory.INCIDENT_REPORT


def days_since(reviewed_on: date, as_of: date) -> int:
    return (as_of - reviewed_on).days


def is_stale(
    category: DocumentCategory | str,
    last_reviewed_at: date,
    as_of: date,
    threshold_days: int = STALE_THRESHOLD_DAYS,
) -> bool:
    if is_review_exempt(category):
        return False
    return days_since(last_reviewed_at, as_of) > threshold_days
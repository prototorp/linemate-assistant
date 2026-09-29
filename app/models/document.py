from dataclasses import dataclass
from datetime import date
 
from app.core import rules
from app.core.config import STALE_THRESHOLD_DAYS
from app.models.enums import DocumentCategory
 
 
@dataclass
class Document:
    id: int
    title: str
    category: DocumentCategory
    body: str
    owner_id: int
    last_reviewed_at: date
 
    def days_since_reviewed(self, as_of: date) -> int:
        return rules.days_since(self.last_reviewed_at, as_of)
 
    def is_stale(self, as_of: date, threshold_days: int = STALE_THRESHOLD_DAYS) -> bool:
        return rules.is_stale(self.category, self.last_reviewed_at, as_of, threshold_days)
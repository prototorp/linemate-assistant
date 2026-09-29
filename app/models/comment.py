from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Comment:
    id: int
    ticket_id: int
    author_id: int
    body: str
    created_at: datetime = field(default_factory=datetime.now)
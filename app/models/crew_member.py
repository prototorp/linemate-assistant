from dataclasses import dataclass
 
from app.models.enums import Station
 
 
@dataclass
class CrewMember:
    id: int
    name: str
    station: Station
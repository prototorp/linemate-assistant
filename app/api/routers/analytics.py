from fastapi import APIRouter, Depends

from app.api.deps import get_knowledge_base_service
from app.api.schemas import WorkloadReport
from app.api.security import require_api_key
from app.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/analytics", tags=["analytics"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=WorkloadReport, summary="Station workload distribution")
@router.get("/workload", response_model=WorkloadReport, include_in_schema=False)
def station_workload(service: KnowledgeBaseService = Depends(get_knowledge_base_service)):
    """Station Workload Distribution: open ticket volume by station and priority, a priority-weighted
    load score, and which stations are more than one standard deviation above the mean."""
    return WorkloadReport(**service.get_station_workload_report())
from fastapi import APIRouter, Depends

from app.api.deps import get_knowledge_base_service
from app.api.schemas import CrewMemberOut
from app.api.security import require_api_key
from app.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/crew", tags=["crew"], dependencies=[Depends(require_api_key)])


@router.get("", response_model=list[CrewMemberOut])
def list_crew(service: KnowledgeBaseService = Depends(get_knowledge_base_service)):
    return [CrewMemberOut.model_validate(c) for c in service.get_all_crew()]
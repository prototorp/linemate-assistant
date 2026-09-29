"""Small deterministic datasets shared by the tests. Plain functions (no pytest, no files), so the
tests never depend on data/*.csv or on today's real date.

Dataset design (TODAY = 2026-09-28, stale threshold 90 days):
  documents  d1 fresh Grill SOP | d2 stale Prep SOP | d3 old Incident Report (exempt from staleness)
             d4 Recipe reviewed exactly 90 days ago (NOT stale) | d5 Onboarding reviewed 91 days ago (stale)
  tickets    t2 + t3 are the only open station mismatches; t4/t7 are closed, t5 has no document,
             t6 points at a document that does not exist
"""

from datetime import date, datetime, timedelta

from app.models import (
    CrewMember, Document, DocumentCategory, Station, Ticket, TicketPriority, TicketStatus,
)

TODAY = date(2026, 9, 28)


def days_ago(n: int) -> date:
    return TODAY - timedelta(days=n)


def make_crew() -> list[CrewMember]:
    return [
        CrewMember(1, "Gia Grill", Station.GRILL),
        CrewMember(2, "Gus Grill", Station.GRILL),
        CrewMember(3, "Pia Prep", Station.PREP),
        CrewMember(4, "Pat Pastry", Station.PASTRY),
        CrewMember(5, "Fay Front", Station.FRONT_OF_HOUSE),
    ]


def make_documents() -> list[Document]:
    return [
        Document(1, "Steak Guide", DocumentCategory.SOP, "Rest steaks 5 minutes.", 1, days_ago(10)),
        Document(2, "Cooler SOP", DocumentCategory.SOP, "Hold cold food at 41F.", 3, days_ago(200)),
        Document(3, "Cooler Incident", DocumentCategory.INCIDENT_REPORT, "Compressor failed.", 3, days_ago(400)),
        Document(4, "Brulee Recipe", DocumentCategory.RECIPE, "Bake in a water bath.", 4, days_ago(90)),
        Document(5, "New Cook Week One", DocumentCategory.ONBOARDING, "Day 1 is the tour.", 3, days_ago(91)),
    ]


def make_tickets() -> list[Ticket]:
    def t(id, title, priority, status, assignee, doc):
        return Ticket(id, title, priority, assignee, status, doc, datetime(2026, 9, 20, 9, 0))

    P, S = TicketPriority, TicketStatus
    return [
        t(1, "Steak chart faded", P.LOW, S.OPEN, 1, 1),            # Grill on a Grill doc: fine
        t(2, "Cooler log gaps", P.HIGH, S.OPEN, 1, 2),             # Grill on a Prep doc: MISMATCH
        t(3, "Cooler door gasket", P.CRITICAL, S.IN_PROGRESS, 5, 2),  # FOH on a Prep doc: MISMATCH
        t(4, "Old cooler issue", P.MEDIUM, S.RESOLVED, 5, 2),      # mismatch but Resolved: excluded
        t(5, "Drain slow", P.MEDIUM, S.OPEN, 1, None),             # no related document: excluded
        t(6, "Ghost doc ref", P.LOW, S.OPEN, 2, 999),              # dangling document: excluded
        t(7, "Closed steak issue", P.LOW, S.CLOSED, 4, 1),         # mismatch but Closed: excluded
        t(8, "Cooler shelf labels", P.HIGH, S.OPEN, 3, 2),         # Prep on a Prep doc: fine
    ]
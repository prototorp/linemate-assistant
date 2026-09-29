"""Business logic queries: Stale Documentation and Station Ownership Mismatch"""

from app.analytics.queries import find_stale_documents, find_station_mismatches
from tests.factories import TODAY, make_crew, make_documents, make_tickets


def test_stale_documents_exclude_incident_reports_and_boundaries():
    stale = find_stale_documents(make_documents(), TODAY, threshold_days=90)
    # d2 (200 days) and d5 (91 days) are stale; d4 (exactly 90) is not; d3 is an old Incident Report
    assert [item.document.id for item in stale] == [2, 5]


def test_stale_documents_sorted_most_overdue_first():
    days = [item.days_since_reviewed for item in find_stale_documents(make_documents(), TODAY)]
    assert days == sorted(days, reverse=True) == [200, 91]


def test_stale_threshold_is_configurable():
    ids = [i.document.id for i in find_stale_documents(make_documents(), TODAY, threshold_days=30)]
    assert ids == [2, 5, 4]


def test_mismatches_are_open_tickets_assigned_outside_the_owning_station():
    mismatches = find_station_mismatches(make_tickets(), make_documents(), make_crew())
    assert sorted(m.ticket.id for m in mismatches) == [2, 3]


def test_mismatch_reports_both_stations():
    by_ticket = {m.ticket.id: m for m in find_station_mismatches(make_tickets(), make_documents(), make_crew())}
    assert by_ticket[2].assignee.station.value == "Grill"
    assert by_ticket[2].owner.station.value == "Prep"
    assert by_ticket[3].assignee.station.value == "Front of House"


def test_resolved_closed_undocumented_and_dangling_tickets_are_ignored():
    ids = {m.ticket.id for m in find_station_mismatches(make_tickets(), make_documents(), make_crew())}
    assert ids.isdisjoint({4, 5, 6, 7})  # resolved, no doc, dangling doc, closed


def test_same_station_different_person_is_not_a_mismatch():
    # ticket 8: Pia (Prep) on a doc owned by Pia's station; and Gus vs Gia are both Grill
    ids = {m.ticket.id for m in find_station_mismatches(make_tickets(), make_documents(), make_crew())}
    assert 8 not in ids and 1 not in ids
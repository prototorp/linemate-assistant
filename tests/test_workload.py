"""Station Workload Distribution (pandas + numpy).

Expected numbers for the shared dataset (open = Open + In-Progress):
  Grill  4 tickets (Low, High, Medium, Low)  load 1+3+2+1 = 7
  Pastry 0 tickets                            load 0
  Prep   1 ticket  (High)                     load 3
  FOH    1 ticket  (Critical)                 load 4
  total 6 tickets, total load 14, mean 3.5, population std 2.5 -> only Grill (7 > 6.0) is overloaded
"""

from app.analytics.workload import compute_station_workload
from app.models import CrewMember, Station, Ticket, TicketPriority, TicketStatus
from tests.factories import make_crew, make_tickets


def _by_station(report):
    return {row["station"]: row for row in report["stations"]}


def test_counts_and_priority_matrix():
    report = compute_station_workload(make_tickets(), make_crew())
    stations = _by_station(report)
    assert report["total_open_tickets"] == 6
    assert stations["Grill"]["open_ticket_count"] == 4
    assert stations["Grill"]["by_priority"] == {"Low": 2, "Medium": 1, "High": 1, "Critical": 0}
    assert stations["Prep"]["by_priority"]["High"] == 1
    assert stations["Front of House"]["by_priority"]["Critical"] == 1


def test_stations_with_no_open_tickets_still_appear():
    stations = _by_station(compute_station_workload(make_tickets(), make_crew()))
    assert stations["Pastry"]["open_ticket_count"] == 0
    assert stations["Pastry"]["load_score"] == 0.0


def test_load_score_is_priority_weighted():
    stations = _by_station(compute_station_workload(make_tickets(), make_crew()))
    assert stations["Grill"]["load_score"] == 7.0
    assert stations["Front of House"]["load_score"] == 4.0  # one Critical outweighs the Prep High


def test_shares_and_overload_flag():
    report = compute_station_workload(make_tickets(), make_crew())
    stations = _by_station(report)
    assert report["mean_load_score"] == 3.5
    assert report["std_load_score"] == 2.5
    assert stations["Grill"]["load_share_pct"] == 50.0
    assert stations["Grill"]["ticket_share_pct"] == 66.7
    assert [s for s, row in stations.items() if row["is_overloaded"]] == ["Grill"]


def test_resolved_and_closed_tickets_are_not_counted():
    report = compute_station_workload(make_tickets(), make_crew())
    assert report["total_open_tickets"] == 6  # 8 tickets total, 2 are Resolved/Closed


def test_unknown_assignee_is_reported_not_silently_dropped():
    tickets = [Ticket(1, "Orphan", TicketPriority.HIGH, 99, TicketStatus.OPEN)]
    report = compute_station_workload(tickets, make_crew())
    assert report["unknown_station_open_tickets"] == 1
    assert report["total_open_tickets"] == 0


def test_no_tickets_does_not_crash_and_flags_nothing():
    report = compute_station_workload([], make_crew())
    assert report["total_open_tickets"] == 0
    assert not any(row["is_overloaded"] for row in report["stations"])
    assert all(row["load_share_pct"] == 0.0 for row in report["stations"])


def test_perfectly_even_load_flags_nobody():
    crew = [CrewMember(i, f"c{i}", s) for i, s in enumerate(Station, start=1)]
    tickets = [Ticket(i, "t", TicketPriority.MEDIUM, i, TicketStatus.OPEN) for i in range(1, 5)]
    report = compute_station_workload(tickets, crew)
    assert not any(row["is_overloaded"] for row in report["stations"])
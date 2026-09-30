"""Station Workload Distribution (pandas + numpy).

Question: how is OPEN ticket volume spread across stations and priority levels, and which
stations carry a disproportionate share?

Approach:
  * left-merge tickets onto crew so every ticket gets a station (unknown assignees are counted)
  * keep Open + In-Progress tickets
  * crosstab station x priority for the volume matrix
  * load score = sum of priority weights (Low 1, Medium 2, High 3, Critical 4), so one
    Critical ticket counts for more than one Low ticket
  * a station is overloaded when its load score is more than one standard deviation above
    the mean of ALL stations (stations with zero open tickets are included in that mean)
"""

import numpy as np
import pandas as pd

from app.core.config import OPEN_STATUSES, PRIORITY_WEIGHT
from app.models import CrewMember, Station, Ticket, TicketPriority

PRIORITY_ORDER = [p.value for p in TicketPriority]
STATION_ORDER = [s.value for s in Station]


def compute_station_workload(tickets: list[Ticket], crew: list[CrewMember]) -> dict:
    tickets_df = pd.DataFrame(
        [
            {
                "priority": t.priority.value,
                "status": t.status.value,
                "assignee_id": t.assignee_id,
            }
            for t in tickets
        ],
        columns=["priority", "status", "assignee_id"],
    )
    crew_df = pd.DataFrame(
        [{"assignee_id": c.id, "station": c.station.value} for c in crew],
        columns=["assignee_id", "station"],
    )

    open_df = tickets_df[tickets_df["status"].isin(OPEN_STATUSES)]
    merged = open_df.merge(crew_df, on="assignee_id", how="left")

    unknown_station = int(merged["station"].isna().sum())
    known = merged.dropna(subset=["station"]).copy()
    known["weight"] = known["priority"].map(PRIORITY_WEIGHT)

    # station x priority matrix, with every station and priority present even when zero
    matrix = (
        pd.crosstab(known["station"], known["priority"])
        .reindex(index=STATION_ORDER, columns=PRIORITY_ORDER, fill_value=0)
    )
    counts = matrix.sum(axis=1).to_numpy(dtype=float)
    load = (matrix * pd.Series(PRIORITY_WEIGHT)[PRIORITY_ORDER]).sum(axis=1).to_numpy(dtype=float)

    total_load = float(load.sum())
    total_count = float(counts.sum())
    load_share = load / total_load * 100 if total_load > 0 else np.zeros_like(load)
    count_share = counts / total_count * 100 if total_count > 0 else np.zeros_like(counts)
    mean_load = float(load.mean())
    std_load = float(load.std())  # population std over the four stations
    overloaded = load > (mean_load + std_load) if std_load > 0 else np.zeros(len(load), dtype=bool)

    stations = [
        {
            "station": station,
            "open_ticket_count": int(counts[i]),
            "by_priority": {p: int(matrix.loc[station, p]) for p in PRIORITY_ORDER},
            "load_score": float(load[i]),
            "ticket_share_pct": round(float(count_share[i]), 1),
            "load_share_pct": round(float(load_share[i]), 1),
            "is_overloaded": bool(overloaded[i]),
        }
        for i, station in enumerate(STATION_ORDER)
    ]

    return {
        "stations": stations,
        "total_open_tickets": int(total_count),
        "unknown_station_open_tickets": unknown_station,
        "mean_load_score": round(mean_load, 2),
        "std_load_score": round(std_load, 2),
        "overload_rule": "load_score > mean + 1 standard deviation across all stations",
    }
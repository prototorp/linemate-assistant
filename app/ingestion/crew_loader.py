"""Loads crew.csv into CrewMember objects."""

import csv
import logging
from pathlib import Path

from app.core.exceptions import CrewLoadError
from app.models import CrewMember, Station

log = logging.getLogger("linemate.ingestion")


def load_crew_from_csv(csv_path: str | Path) -> list[CrewMember]:
    crew: list[CrewMember] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                try:
                    member = CrewMember(
                        id=int(row["id"]),
                        name=row["name"].strip(),
                        station=Station(row["station"].strip()),
                    )
                except (ValueError, KeyError) as exc:
                    raise CrewLoadError(f"crew row {row.get('id')}: {exc}") from exc
            except CrewLoadError as exc:
                log.warning("SKIPPED %s", exc)
                continue
            crew.append(member)
    return crew
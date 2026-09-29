from app.core import rules
from app.models import DocumentCategory
from tests.factories import TODAY, days_ago


def test_exactly_at_threshold_is_not_stale():
    assert rules.is_stale(DocumentCategory.SOP, days_ago(90), TODAY, 90) is False


def test_one_day_past_threshold_is_stale():
    assert rules.is_stale(DocumentCategory.SOP, days_ago(91), TODAY, 90) is True


def test_incident_reports_never_go_stale():
    assert rules.is_stale(DocumentCategory.INCIDENT_REPORT, days_ago(5000), TODAY) is False


def test_accepts_category_as_plain_string():
    assert rules.is_stale("SOP", days_ago(200), TODAY) is True
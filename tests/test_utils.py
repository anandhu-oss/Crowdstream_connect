import logging
import pytest

from src.utils.logging import get_logger, log_event
from src.utils.timestamps import format_hms, overlap, parse_time, srt_time, vtt_time


def test_parse_time():
    assert parse_time("00:00:01.500") == 1.5 and parse_time("01:02:03.250") == 3723.25 and parse_time("00:00:05") == 5.0
    assert parse_time("00:00:01,5") == 1.5 and parse_time("00:00:00.040") == pytest.approx(0.04)
    for bad in ("1.5", "00:61:00.000", "abc", "", None, "00:00"):
        with pytest.raises(ValueError): parse_time(bad)


def test_formatting_roundtrip():
    assert format_hms(3723.25) == "01:02:03.250" and srt_time(1.5) == "00:00:01,500" and vtt_time(0) == "00:00:00.000"
    assert parse_time(format_hms(12.345)) == pytest.approx(12.345)
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError): format_hms(bad)


def test_overlap():
    assert overlap(0, 5, 3, 8) == 2 and overlap(0, 1, 2, 3) == 0 and overlap(0, 10, 2, 3) == 1


def test_log_event_refuses_content(caplog):
    log = get_logger("test_logger")
    with caplog.at_level(logging.INFO, logger="test_logger"):
        log_event(log, "job_done", job_id="a1", seconds=1.5)
    assert "job_id=a1" in caplog.text
    for bad in ({"text": "hi"}, {"Transcript": "x"}, {"audio": b""}):
        with pytest.raises(ValueError): log_event(log, "x", **bad)

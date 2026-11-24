"""Logging that cannot leak content: only ids, times, counts and statuses are accepted."""
import logging

FORBIDDEN = {"text", "audio", "transcript", "translation", "caption", "hypothesis", "reference", "content"}


def get_logger(name: str = "connect_ai") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler(); h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(h); log.setLevel(logging.INFO)
    return log


def log_event(log: logging.Logger, event: str, **fields):
    """log_event(log, 'job_done', job_id='ab12', seconds=3.2). Raises if a field could carry audio or text."""
    bad = FORBIDDEN & {k.lower() for k in fields}
    if bad:
        raise ValueError(f"log fields must not carry content: {sorted(bad)}")
    log.info("%s %s", event, " ".join(f"{k}={v}" for k, v in fields.items()))

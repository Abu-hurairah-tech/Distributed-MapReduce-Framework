"""Workloads: Word Frequency (``wordfreq``) and Inverted Index (``invindex``)."""
from __future__ import annotations

from mapreduce.api import Job

from . import inverted_index, word_frequency

_JOBS = {
    word_frequency.JOB.name: word_frequency.JOB,
    inverted_index.JOB.name: inverted_index.JOB,
}

JOB_NAMES = tuple(_JOBS)  # ("wordfreq", "invindex")


def get_job(name: str) -> Job:
    """Return the :class:`Job` registered under ``name``."""
    try:
        return _JOBS[name]
    except KeyError:
        raise ValueError(f"unknown job {name!r}; choose from {', '.join(JOB_NAMES)}") from None

"""Inverted Index: for every term, the sorted list of documents containing it.

``doc_id`` is the input file's basename (``key[0]`` of the map record).
"""
from __future__ import annotations

from mapreduce.api import Combiner, Job, Mapper, Reducer

from .word_frequency import tokenize


class InvertedIndexMapper(Mapper):
    def map(self, key, value, ctx):
        doc_id = key[0]
        for term in tokenize(value):
            yield term, doc_id


class InvertedIndexCombiner(Combiner):
    """Emit each unique ``(term, doc_id)`` once."""

    def reduce(self, key, values, ctx):
        for doc_id in sorted(set(values)):
            yield key, doc_id


class InvertedIndexReducer(Reducer):
    def reduce(self, key, values, ctx):
        yield key, ",".join(sorted(set(values)))


JOB = Job(
    name="invindex",
    mapper=InvertedIndexMapper,
    reducer=InvertedIndexReducer,
    combiner=InvertedIndexCombiner,
)

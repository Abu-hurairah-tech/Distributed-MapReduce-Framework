"""Word Frequency: count how often every word occurs in the corpus."""
from __future__ import annotations

import re

from mapreduce.api import Combiner, Job, Mapper, Reducer

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(line: str) -> list:
    return _TOKEN.findall(line.lower())


class WordFrequencyMapper(Mapper):
    def map(self, key, value, ctx):
        for word in tokenize(value):
            yield word, 1


class WordFrequencyCombiner(Combiner):
    def reduce(self, key, values, ctx):
        yield key, sum(values)


class WordFrequencyReducer(Reducer):
    def reduce(self, key, values, ctx):
        yield key, sum(values)


JOB = Job(
    name="wordfreq",
    mapper=WordFrequencyMapper,
    reducer=WordFrequencyReducer,
    combiner=WordFrequencyCombiner,
)

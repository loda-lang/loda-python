# -*- coding: utf-8 -*-

"""Integer sequence model."""

import functools


@functools.total_ordering
class Sequence:
    def __init__(self, id: int, name="", terms=[]):
        self.id = id
        self.name = name
        self.terms = terms

    def __str__(self) -> str:
        return "{}: {}".format(self.id_str(), self.name)

    def __eq__(self, other) -> bool:
        return self.id == other.id and self.terms == other.terms

    def __lt__(self, other) -> bool:
        if self.terms < other.terms:
            return True
        if self.terms == other.terms:
            return self.id < other.id
        return False

    def id_str(self) -> str:
        return "A{:06}".format(self.id)

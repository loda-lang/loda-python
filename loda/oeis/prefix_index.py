"""Prefix index for searching integer sequences."""

import copy
import functools

from .sequence import Sequence


class PrefixIndex:

    class Match:
        def __init__(self, size: int):
            self.prefix_length = 0
            self.start_index = 0
            self.end_index = size  # exclusive
            self.finished_ids = []

    def __init__(self, seqs: list):
        seqs = list(filter(lambda s: len(s.terms) > 0, seqs))
        max_id = functools.reduce(lambda id, s: max(id, s.id), seqs, 0)
        self.__index = sorted(seqs)
        self.__lookup = [0] * (max_id + 1)
        for i in range(len(seqs)):
            id = self.__index[i].id
            self.__lookup[id] = i

    def size(self) -> int:
        return len(self.__index)

    def get(self, id: int) -> Sequence:
        return copy.copy(self.__get(id))

    def __get(self, id: int) -> Sequence:
        return self.__index[self.__lookup[id]]

    def global_match(self) -> Match:
        return PrefixIndex.Match(len(self.__index))

    def refine_match(self, match: Match, term: int) -> bool:
        if match.start_index >= match.end_index:
            return False
        arg = match.prefix_length
        match.prefix_length += 1
        new_start = match.start_index
        while new_start < match.end_index and self.__index[new_start].terms[arg] < term:
            new_start += 1
        while new_start < match.end_index and self.__index[new_start].terms[arg] == term and len(self.__index[new_start].terms) == match.prefix_length:
            match.finished_ids.append(self.__index[new_start].id)
            new_start += 1
        new_end = new_start
        while new_end < match.end_index and self.__index[new_end].terms[arg] == term:
            new_end += 1
        match.start_index = new_start
        match.end_index = new_end
        return new_start < new_end

    def get_match_ids(self, match: Match) -> list:
        ids = [self.__index[i].id for i in range(
            match.start_index, match.end_index)]
        ids.extend(match.finished_ids)
        return sorted(ids)

# -*- coding: utf-8 -*-

"""Prefix index for searching integer sequences."""

import copy
import os.path
import re

from .sequence import Sequence


class PrefixIndex:

    class Match:
        def __init__(self, size: int):
            self.prefix_length = 0
            self.start_index = 0
            self.end_index = size  # exclusive
            self.finished_ids = []

    def __init__(self, path: str):
        self.__path = path
        self.__index = None
        self.__lookup = None

    def size(self) -> int:
        if self.__index is None:
            self.__load()
        return len(self.__index)

    def get(self, id: int):
        if self.__index is None:
            self.__load()
        return copy.copy(self.__get(id))

    def __get(self, id: int):
        return self.__index[self.__lookup[id]]

    def __parse_line(self, line: str, pattern):
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            return None
        match = pattern.match(line)
        if not match:
            raise ValueError("parse error: {}".format(line))
        return match

    def __load(self):
        seqs = []
        # load sequence terms
        stripped = os.path.join(self.__path, "stripped")
        expected_id = 1
        with open(stripped) as file:
            pattern = re.compile("^A([0-9]+) ,([0-9,]+),$")
            for line in file:
                match = self.__parse_line(line, pattern)
                if not match:
                    continue
                id = int(match.group(1))
                if id != expected_id:
                    raise ValueError("unexpected ID: {}".format(line))
                terms_str = match.group(2).split(",")
                terms = [int(t) for t in terms_str]
                seqs.append(Sequence(id, "", terms))
                expected_id += 1
        # load sequence names
        names = os.path.join(self.__path, "names")
        expected_id = 1
        with open(names) as file:
            pattern = re.compile("^A([0-9]+) (.+)$")
            for line in file:
                match = self.__parse_line(line, pattern)
                if not match:
                    continue
                id = int(match.group(1))
                if id != expected_id:
                    raise ValueError("unexpected ID: {}".format(line))
                name = match.group(2)
                seqs[id - 1].name = name
                expected_id += 1
        self.__index = sorted(seqs)
        self.__lookup = [0] * (len(seqs) + 1)
        for i in range(len(seqs)):
            id = self.__index[i].id
            self.__lookup[id] = i

    def global_match(self) -> Match:
        if self.__index is None:
            self.__load()
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

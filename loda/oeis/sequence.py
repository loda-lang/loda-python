"""Integer sequence model."""

import functools
import os.path
import re


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

    def align(self, terms: list, max_offset: int = 5) -> list:
        # check if they agree on prefix already
        min_length = min(len(self.terms), len(terms))
        if self.terms[0:min_length] == terms[0:min_length]:
            return True
        # try to align them
        for offset in range(1, max_offset+1):
            if offset >= min_length:
                break
            agree_pos = True
            agree_neg = True
            for i in range(min_length):
                if i+offset < len(self.terms) and self.terms[i + offset] != terms[i+offset]:
                    agree_pos = False
                if i+offset < len(terms) and self.terms[i] != terms[i+offset]:
                    agree_neg = False
            if agree_pos:
                return terms[offset:]
            if agree_neg:
                result = self.terms[0:offset]
                result.extend(terms)
                return result
        return None

    def load_b_file(self, path: str) -> list:
        terms = []
        if len(path) == 0 or os.path.isdir(path):
            dir = "{:03}".format(self.id//1000)
            asm = "{}.asm".format(self.id_str())
            path = os.path.join(path, dir, asm)
        with open(path) as b_file:
            expected_index = -1
            for line in b_file:
                line = line.strip()
                if len(line) == 0 or line[0] == "#":
                    continue
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError("unexpected line: {}".format(line))
                index = int(fields[0])
                value = int(fields[0])
                if expected_index == -1:
                    expected_index = index
                    if index != expected_index:
                        raise ValueError("unexpected index: {}".format(index))
                terms.append(value)
                expected_index += 1
        terms = self.align(terms)
        if terms is None:
            raise ValueError("error aliging sequences")
        if len(terms) < len(self.terms):
            raise ValueError("too few b-file terms")
            # terms = self.terms
        if terms[0:len(self.terms)] != self.terms:
            raise ValueError("unexpected terms in b-file")
        return terms


def __parse_line(line: str, pattern):
    line = line.strip()
    if len(line) == 0 or line.startswith("#"):
        return None
    match = pattern.match(line)
    if not match:
        raise ValueError("parse error: {}".format(line))
    return match


def __fill_seqs(seqs: list, id: int):
    current_size = len(seqs)
    for i in range(current_size, id+2):
        seqs.append(Sequence(i, "", []))


def load(oeis_path: str) -> list:
    """
    Load sequences from `stripped` from `names` files.
    """
    seqs = []
    # load sequence terms
    stripped = os.path.join(oeis_path, "stripped")
    with open(stripped) as file:
        pattern = re.compile("^A([0-9]+) ,([\\-0-9,]+),$")
        for line in file:
            match = __parse_line(line, pattern)
            if not match:
                continue
            id = int(match.group(1))
            __fill_seqs(seqs, id)
            seqs[id].id = id
            terms_str = match.group(2).split(",")
            seqs[id].terms = [int(t) for t in terms_str]
    # load sequence names
    names = os.path.join(oeis_path, "names")
    with open(names) as file:
        pattern = re.compile("^A([0-9]+) (.+)$")
        for line in file:
            match = __parse_line(line, pattern)
            if not match:
                continue
            id = int(match.group(1))
            __fill_seqs(seqs, id)
            name = match.group(2)
            seqs[id].name = name
    return seqs

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

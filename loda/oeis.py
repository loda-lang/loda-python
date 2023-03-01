# -*- coding: utf-8 -*-

import os.path
from loda.lang import Program


class Sequence:
    def __init__(self, id: int, name="", terms=[]):
        self.id = id
        self.name = name
        self.terms = terms

    def id_str(self) -> str:
        return "A{:06}".format(self.id)


class SequenceCache:
    def __init__(self, path: str, auto_fetch = False):
        self.__path = path
        self.__auto_fetch = auto_fetch
        self.__cache = None

    def __fetch():
        # TODO
        pass

    def __load():
        # TODO
        pass

    def get(self, id: int, use_b_file=False) -> Sequence:
        if self.__cache is None:
            self.__load()
        # TODO
        if use_b_file:
            pass


class ProgramCache:

    def __init__(self, path: str):
        self.__path = path
        self.__cache = {}

    def path(self, id: int) -> str:
        dir = "{:03}".format(id//1000)
        asm = "{}.asm".format(Sequence(id).id_str())
        return os.path.join(self.__path, dir, asm)

    def get(self, id: int) -> Program:
        if id not in self.__cache:
            with open(self.path(id), "r") as file:
                self.__cache[id] = Program(file.read())
        return self.__cache[id]

    def clear(self) -> None:
        self.__cache.clear()

# -*- coding: utf-8 -*-

import os.path

from loda.lang import Program
from .sequence import Sequence


class ProgramCache:

    def __init__(self, path: str):
        self.__path = path
        self.__cache = {}

    def path(self, id: int) -> str:
        dir = "{:03}".format(id//1000)
        asm = "{}.asm".format(Sequence(id).id_str())
        return os.path.join(self.__path, dir, asm)

    def get(self, id: int):
        if id not in self.__cache:
            with open(self.path(id), "r") as file:
                self.__cache[id] = Program(file.read())
        return self.__cache[id]

    def all_ids(self):
        ids = []
        for dir in os.listdir(self.__path):
            child = os.path.join(self.__path, dir)
            if not os.path.isdir(child):
                continue
            for file in os.listdir(child):
                if file.startswith('A') and file.endswith('.asm'):
                    ids.append(int(file[1:7]))
        ids.sort()
        return ids

    def clear(self) -> None:
        self.__cache.clear()

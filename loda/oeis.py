# -*- coding: utf-8 -*-

def getOeisId(id):
    return 'A' + str(id).zfill(6)


def getOeisProgramPath(id):
    return str(int(id/1000)).zfill(3) + '/' + getOeisId(id) + '.asm'

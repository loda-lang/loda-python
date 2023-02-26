# -*- coding: utf-8 -*-

from loda.lang import Operation
import math


def add(a, b):
    if a == None or b == None:
        return None
    return a + b


def sub(a, b):
    if a == None or b == None:
        return None
    return a - b


def trn(a, b):
    if a == None or b == None:
        return None
    return max(a - b, 0)


def mul(a, b):
    if a == None or b == None:
        return None
    return a * b


def div(a, b):
    if a == None or b == None or b == 0:
        return None
    s = 1 if (a < 0) == (b < 0) else -1
    return s * (abs(a) // abs(b))


def dif(a: int, b: int) -> int:
    if a == None or b == None:
        return None
    if b == 0:
        return a
    d = div(a, b)
    return d if a == mul(b, d) else a


def mod(a, b):
    if a == None or b == None or b == 0:
        return None
    return a - mul(b, div(a, b))


def pow(a, b):
    if a == None or b == None:
        return None
    if a == 0:
        if b > 0:
            return 0  # 0^(positive number)
        elif b == 0:
            return 1  # 0^0
        else:
            return None  # 0^(negative number) => inf
    elif a == 1:
        return 1  # 1^x is always 1
    elif a == -1:
        return 1 if mod(b, 2) == 0 else -1  # (-1)^x
    else:
        if b < 0:
            return 0
        else:
            return a**b


def gcd(a, b):
    if a == None or b == None:
        return None
    return math.gcd(a, b)


def bin(n, k):
    if n == None or k == None:
        return None
    # check for negative arguments: https://arxiv.org/pdf/1105.3689.pdf
    sign = 1
    if n < 0:  # Theorem 2.1
        if k >= 0:
            sign = 1 if mod(k, 2) == 0 else -1
            n = sub(k, add(n, 1))
        elif n >= k:
            sign = 1 if mod(sub(n, k), 2) == 0 else -1
            m = n
            n = sub(0, add(k, 1))
            k = sub(m, k)
        else:
            return 0
    if k < 0 or n < k:  # 1.2
        return 0
    if n < mul(k, 2):
        k = sub(n, k)
    # main calculation
    r = 1
    for i in range(0, k):
        r = mul(r, sub(n, i))
        r = div(r, add(i, 1))
        if r == None:
            break
    return mul(sign, r)


def cmp(a, b):
    if a == None or b == None:
        return None
    return 1 if a == b else 0


def min(a, b):
    if a == None or b == None:
        return None
    return a if a < b else b


def max(a, b):
    if a == None or b == None:
        return None
    return a if a > b else b


def calc_arith(t: Operation.Type, a, b):
    if t == Operation.Type.MOV:
        return b
    elif t == Operation.Type.ADD:
        return add(a, b)
    elif t == Operation.Type.SUB:
        return sub(a, b)
    elif t == Operation.Type.TRN:
        return trn(a, b)
    elif t == Operation.Type.MUL:
        return mul(a, b)
    elif t == Operation.Type.DIV:
        return div(a, b)
    elif t == Operation.Type.DIF:
        return dif(a, b)
    elif t == Operation.Type.MOD:
        return mod(a, b)
    elif t == Operation.Type.POW:
        return pow(a, b)
    elif t == Operation.Type.GCD:
        return gcd(a, b)
    elif t == Operation.Type.BIN:
        return bin(a, b)
    elif t == Operation.Type.CMP:
        return cmp(a, b)
    elif t == Operation.Type.MIN:
        return min(a, b)
    elif t == Operation.Type.MAX:
        return max(a, b)
    else:
        raise ValueError('operation type not arithmetic: ' + t)

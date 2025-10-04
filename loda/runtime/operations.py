# -*- coding: utf-8 -*-

"""Evaluate operations."""

from loda.lang import Operation
import math


def add(a, b):
    """Addition."""
    if a == None or b == None:
        return None
    return a + b

def sub(a, b):
    """Subtraction."""
    if a == None or b == None:
        return None
    return a - b

def trn(a, b):
    """Truncated Subtraction."""
    if a == None or b == None:
        return None
    return max(a - b, 0)

def mul(a, b):
    """Multiplication."""
    if a == None or b == None:
        return None
    return a * b

def div(a, b):
    """Division."""
    if a == None or b == None or b == 0:
        return None
    s = 1 if (a < 0) == (b < 0) else -1
    return s * (abs(a) // abs(b))

def dif(a, b):
    """Conditional Division."""
    if a == None or b == None:
        return None
    if b == 0:
        return a
    d = div(a, b)
    return d if a == mul(b, d) else a

def dir(a, b):
    """Repeated conditional division."""
    if a is None or b is None:
        return None
    aa = a
    while True:
        r = dif(aa, b)
        if abs(r) == abs(aa):
            break
        aa = r
    return aa

def mod(a, b):
    """Modulus (Remainder)."""
    if a == None or b == None or b == 0:
        return None
    return a - mul(b, div(a, b))

def pow(a, b):
    """Power."""
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
    """Greatest Common Divisor."""
    if a == None or b == None:
        return None
    return math.gcd(a, b)

def lex(a, b):
    """Largest exponent: returns the largest k such that b^k divides a (C++ semantics)."""
    if a is None or b is None:
        return None
    if b == 0 or abs(b) == 1:
        return 0
    r = 0
    aa = abs(a)
    bb = abs(b)
    while True:
        aaa = dif(aa, bb)
        if aaa == aa:
            break
        aa = aaa
        r += 1
    return r

def bin(n, k):
    """Binomial Coefficient."""
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
    for i in range(k):
        r = mul(r, sub(n, i))
        r = div(r, add(i, 1))
        if r == None:
            break
    return mul(sign, r)

def fac(n, k):
    """Falling and rising factorial."""
    if n is None or k is None:
        return None
    d = 1
    res = 1
    if k < 0:
        k = -k
        d = -1
    for i in range(k):
        res *= n
        if res == 0:
            return 0
        n += d
    return res

def log(a, b):
    """Discrete logarithm: returns the integer part of log_b(a)."""
    if a is None or b is None or a < 1 or b < 2:
        return None
    if a == 1:
        return 0
    m = 1
    res = 0
    while m < a:
        m *= b
        res += 1
    return res if m == a else res - 1

def nrt(a, b):
    """n-th root: returns the integer part of the b-th root of a."""
    if a is None or b is None or a < 0 or b < 1:
        return None
    if a == 0 or a == 1 or b == 1:
        return a
    r = 1
    l = 0
    h = a
    while l < h:
        m = div(add(l, h), 2)
        p = pow(m, b)
        if p == a:
            return m
        if p < a:
            l = m
        else:
            h = m
        if r == m:
            break
        r = m
    return r

def dgs(a, b):
    """Digit sum in base b."""
    if a is None or b is None or b < 2:
        return None
    sign = -1 if a < 0 else 1
    aa = abs(a)
    r = 0
    while aa > 0:
        r += aa % b
        aa //= b
    return sign * r

def dgr(a, b):
    """Digital root in base b."""
    if a is None or b is None or b < 2:
        return None
    if a == 0:
        return 0
    sign = -1 if a < 0 else 1
    aa = abs(a)
    return sign * (1 + ((aa - 1) % (b - 1)))

def equ(a, b):
    """Equality."""
    if a == None or b == None:
        return None
    return 1 if a == b else 0

def neq(a, b):
    """Inequality."""
    if a == None or b == None:
        return None
    return 1 if a != b else 0

def leq(a, b):
    """Less or equal."""
    if a == None or b == None:
        return None
    return 1 if a <= b else 0

def geq(a, b):
    """Greater or equal."""
    if a == None or b == None:
        return None
    return 1 if a >= b else 0

def min(a, b):
    """Minimum."""
    if a == None or b == None:
        return None
    return a if a < b else b

def max(a, b):
    """Maximum."""
    if a == None or b == None:
        return None
    return a if a > b else b

def ban(a, b):
    """Bitwise AND."""
    if a is None or b is None:
        return None
    return a & b

def bor(a, b):
    """Bitwise OR."""
    if a is None or b is None:
        return None
    return a | b

def bxo(a, b):
    """Bitwise XOR."""
    if a is None or b is None:
        return None
    return a ^ b

def exec_arithmetic(t: Operation.Type, a, b):
    """Execute an arithmetic operation."""
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
    elif t == Operation.Type.DIR:
        return dir(a, b)
    elif t == Operation.Type.MOD:
        return mod(a, b)
    elif t == Operation.Type.POW:
        return pow(a, b)
    elif t == Operation.Type.GCD:
        return gcd(a, b)
    elif t == Operation.Type.LEX:
        return lex(a, b)
    elif t == Operation.Type.BIN:
        return bin(a, b)
    elif t == Operation.Type.FAC:
        return fac(a, b)
    elif t == Operation.Type.LOG:
        return log(a, b)
    elif t == Operation.Type.NRT:
        return nrt(a, b)
    elif t == Operation.Type.DGS:
        return dgs(a, b)
    elif t == Operation.Type.DGR:
        return dgr(a, b)
    elif t == Operation.Type.EQU:
        return equ(a, b)
    elif t == Operation.Type.NEQ:
        return neq(a, b)
    elif t == Operation.Type.LEQ:
        return leq(a, b)
    elif t == Operation.Type.GEQ:
        return geq(a, b)
    elif t == Operation.Type.MIN:
        return min(a, b)
    elif t == Operation.Type.MAX:
        return max(a, b)
    elif t == Operation.Type.BAN:
        return ban(a, b)
    elif t == Operation.Type.BOR:
        return bor(a, b)
    elif t == Operation.Type.BXO:
        return bxo(a, b)
    else:
        raise ValueError("operation type not arithmetic: {}".format(t))

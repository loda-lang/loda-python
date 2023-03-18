This Python package allows you to read and write LODA programs, to evaluate
them to integer sequences, to search for matches in the
[OEIS](https://www.oeis.org/) database,
and to use machine learning from [Tensorflow](https://www.tensorflow.org/)
to generate new integer sequence programs.

## Installation

You need Python 3.7 or higher. To install the dependencies for LODA, run these commands:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Getting Started

LODA programs are stored in `*.asm` files. Below you can find the example program `fibonacci.asm` which
computes the Fibonacci numbers. For a comprehensive overview of the language, see the [LODA Language Specification](https://loda-lang.org/spec/).

```asm
; A000045: Fibonacci numbers.
mov $3,1
lpb $0
  sub $0,1
  mov $2,$1
  add $1,$3
  mov $3,$2
lpe
mov $0,$1
```

Check out the [sub-modules](#header-submodules) for working with LODA programs.

## Development

To execute the tests, run the following command:

```bash
nose2 tests -v
```

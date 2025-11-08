# LODA Python

This Python package contains an implementation of the [LODA Language](https://loda-lang.org/):
an assembly language and computational model for finding integer sequence programs.

This Python package allows you to read and write LODA programs, to evaluate
them to integer sequences, and to search for matches in the
[OEIS](https://www.oeis.org/) database.

## Getting Started

You need Python 3.7 or higher. To install the dependencies for LODA, run these commands:

```bash
python3 -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
```

To execute the tests, run the following command:

```bash
nose2 tests -v
```

Check out [sample.py](sample.py) and the [documentation](https://loda-lang.org/loda-python/) to find out how to use the LODA Python package.

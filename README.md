# LODA Python Module

This Python module contains an implementation of the [LODA Language](https://loda-lang.org/). You can use it to read, write and evaluate LODA programs to integer sequences, and to use machine learning to find new programs.

## Getting Started

Install the dependencies for the LODA Python module:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

To execute the tests, run the following command:

```bash
nose2 tests -v
```

Check out [sample.py](sample.py) and the [documentation](https://loda-lang.org/loda-python/) to find out how to use the LODA Python module.

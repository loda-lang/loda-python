init:
	pip install -r requirements.txt

test:
	nose2 tests -v

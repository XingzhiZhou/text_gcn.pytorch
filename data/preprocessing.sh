#!/usr/bin/bash

PYTHON="/usr/local/bin/python-csd"

for i in R8 R52 ohsumed mr; do
	${PYTHON} remove_words.py $i
	${PYTHON}  build_graph.py $i
done


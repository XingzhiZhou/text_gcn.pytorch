#!/usr/bin/bash

PYTHON="/usr/local/bin/python-csd"
for dataset in ohsumed mr 20ng; do
	for num_mlp_layer in $(seq 1 4)
	do
		for i in $(seq 1 10)
			do
				${PYTHON} train.py ${dataset} --num_mlp_layer ${num_mlp_layer}
				${PYTHON} train.py ${dataset} --num_mlp_layer ${num_mlp_layer} --train_eps
			done
	done
done
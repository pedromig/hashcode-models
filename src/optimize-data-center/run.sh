#!/usr/bin/env sh

mkdir -p output
for f in input/*; do
	file=$(basename $f)
	echo "FILE: $f"
	./src/main.py < $f > output/${file%.*}.out
	echo
done

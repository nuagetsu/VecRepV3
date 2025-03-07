#!/bin/bash
for i in {1..20}
do
    echo $i
    python src/scripts/GenerateShapes.py
done

#!/bin/bash

parallel --citation

# shellcheck disable=SC2162
while read p; do
  sem -j 2 python3 ../main.py $p &
done <parameters.txt
sem --wait
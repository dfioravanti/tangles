#!/bin/bash
for i in `seq 1 10`;
do
    python main.py -s $i --id 51 --no_plots &
done   
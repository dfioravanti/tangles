#!/bin/bash
for i in `seq 1 10`;
do
    python main.py -s $i --id 14 --no_plots &
done   
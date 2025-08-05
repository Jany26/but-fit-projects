#!/bin/bash

# Ján Maťufka / xmatuf00 / 222124
# Part of PRL 2023/24 2nd project - Game of Life using MPI

if [ $# -ne 2 ]; then
    echo "No input file given or no steps (number of iterations) given."
    echo "Usage: $0 <gamefile> <steps>"
    exit 1
fi

gamefile="$1"
steps="$2"

if [ ! -f "$gamefile" ]; then
    echo "File '$gamefile' not found."
    exit 1
fi

lines=$(wc -l < "$gamefile")

mpic++ --prefix /usr/local/share/OpenMPI -o life life.cpp
mpirun --oversubscribe --prefix /usr/local/share/OpenMPI -np $lines life $gamefile $steps

rm -f life

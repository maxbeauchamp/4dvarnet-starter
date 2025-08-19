#!/bin/bash

while true
do
    python preprocessing_CROSCIM_multires.py
    echo "Le script a crashé avec le code $?. Relance dans 5 secondes..."
    sleep 5
done


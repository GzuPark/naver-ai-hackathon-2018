#!/bin/bash
echo "Kind: $1"
echo "Session: $2"
echo "Account: $name"

if [ $name -eq ""]; then
  name="GzuPark"
fi

nsml dataset board $1 &> $1_submitted_total &&
nsml model ls $name/movie_phase1/$2 &> $1_$2 &&
python3 helper.py --account $name --kind $1 --session $2 &&
rm $1_submitted_total &&
rm $1_$2;

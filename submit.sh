#!/bin/bash
echo "Kind: $1"
echo "Session: $2"
echo "Epochs Start: $3"
echo "Epochs End: $4"
echo "Account: $name"

if [ $name -eq ""]; then
  name="GzuPark"
fi

e=`expr $4 - 1`
for i in $(seq $3 $e)
  do
    nsml submit $name/$1/$2 $i;
    echo Done. $name/$1/$2 $i;
done

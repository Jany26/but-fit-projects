#!/bin/bash

# test projektu 3 IZP - 2019
# autor: ihynek

if [ "$1" = "" ] ; then
  path="./"
else
  path=$1
fi

proj="proj3"

echo "gcc -std=c99 -pedantic -Wall -Wextra -o "$path"$proj "$path"$proj.c"
gcc -std=c99 -pedantic -Wall -Wextra -o "$path"$proj "$path"$proj.c
echo "---------------------
"

echo "$path$proj --test bludiste.txt"
expected=`echo -e "Valid\n"`
result=`timeout 5s "$path$proj" --test bludiste.txt`
if [ "$result" = "$expected" ] ; then
  echo "OK"
else 
  echo " ----- EXPECTED:"
  echo "$expected"
  echo " ----- RESULT:"
  echo "$result"
fi
echo "---------------------"
timeout 5s valgrind "$path$proj" --test bludiste.txt 2>&1 | grep 'in use at exit\|ERROR SUMMARY'
echo ""

echo "$path$proj --lpath 3 4 bludiste.txt"
expected=`echo -e "3,4\n3,3\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n1,4\n1,3\n1,2\n1,1\n"`
result=`timeout 5s "$path$proj" --lpath 3 4 bludiste.txt`
if [ "$result" = "$expected" ] ; then
  echo "OK"
else
  echo " ----- EXPECTED:"
  echo "$expected"
  echo " ----- RESULT:"
  echo "$result"
fi
echo "---------------------"
timeout 5s valgrind "$path$proj" --lpath 3 4 bludiste.txt 2>&1 | grep 'in use at exit\|ERROR SUMMARY'
echo ""

echo "$path$proj --rpath 3 4 bludiste.txt"
expected=`echo -e "3,4\n3,3\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n"`
result=`timeout 5s "$path$proj" --rpath 3 4 bludiste.txt`
if [ "$result" = "$expected" ] ; then
  echo "OK"
else
  echo " ----- EXPECTED:"
  echo "$expected"
  echo " ----- RESULT:"
  echo "$result"
fi
echo "---------------------"
timeout 5s valgrind "$path$proj" --rpath 3 4 bludiste.txt 2>&1 | grep 'in use at exit\|ERROR SUMMARY'
echo ""


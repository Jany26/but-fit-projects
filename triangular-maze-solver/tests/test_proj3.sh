#!/bin/bash
# Author: Martin Sakin, isakin<at>fit.vutbr.cz
# Date: 2019-12-11
# Mini test for the defense for IZP proj3
# The result of this script does not guarantee points!!!

END=false
TMP=tmpfile.x
REF=tmpfile2.x
M1=tmp_small.txt
M2=tmp_assignment.txt
M3=tmp_wrong_smaller.txt
M4=tmp_wrong_bigger.txt
M5=tmp_wrong_values.txt


create_mazes()
{
echo "3 4
1 4 4 2
1 4 4 0
1 4 4 4" > $M1

echo "6 7
1 4 4 2 5 0 6
1 4 4 0 4 0 2
1 0 4 0 4 6 1
1 2 7 1 0 4 2
3 1 4 2 3 1 2
4 2 5 0 4 2 5" > $M2

echo "6 7
1 4 4 2 5 0 6
1 4 4 0 4 0 2
1 0 4 0 4 6 1
1 2 7 1 0 4 2
3 1 4 2 3 1 2
4 2 2" > $M3

echo "6 7
1 4 4 2 5 0 6
1 4 4 0 4 0 2
1 0 4 0 4 6 1
1 2 7 1 0 4 2
3 1 4 2 3 1 2
4 2 5 0 4 2 5
1 4 4 2 5 0 6
1 4 4 0 4 0 2
1 0 4 0 4 6 1
1 2 7 1 0 4 2
3 1 4 2 3 1 2
4 2 5 0 4 2 5" > $M4

echo "6 7
1 4 4 9 5 0 6
1 4 -4 0 4 0 2
1 0 4 0 4 6 1
1 2 7 10 0 4 2
3 1 4 2 3 1 2
4 2 5 0 4 -8 5" > $M5
}


delete_mazes_exit()
{
	rm -f $M1 $M2 $M3 $M4 $M5
	exit
}

print_ok()
{
	echo -e " - \e[1;32mOK\e[0m"
}

print_fail()
{
	echo -e " - \e[1;31mFAIL!\e[0m"
}


test_1_help()
{
	if [ ! -f "proj3" ] ; then
		echo " ! Missing proj3"
		exit
	fi

	./proj3 --help >> $TMP 2>> $TMP
	if [ ! "$?" = 0 ] ; then
		echo " ! --help is not wrong, jesus."
		END=true
	fi
	if [ ! -s $TMP ] ; then
		echo " ! And where is the help?"
		END=true
	fi

	rm -f $TMP
	if [ "$END" = true ] ; then
		exit;
	else
		print_ok
	fi
}


test_2_test()
{
	./proj3 --test $M1
	if [ ! "$?" = 0 ] ; then
		echo " ! This map is OK, return value should be zero"
		END=true
	fi

	./proj3 --test $M2
	if [ ! "$?" = 0 ] ; then
		echo " ! This map is OK, return value should be zero"
		END=true
	fi

	./proj3 --test $M3
	if [ "$?" = 0 ] ; then
		echo " ! Map is smaller, return value should not be zero"
		END=true
	fi

	./proj3 --test $M4
	if [ "$?" = 0 ] ; then
		echo " ? Map is bigger, are you sure?"
	fi

	./proj3 --test $M5
	if [ "$?" = 0 ] ; then
		echo " ! Map contains incorrect values"
		END=true
	fi

	rm -f $TMP
	if [ "$END" = true ] ; then
		print_fail
	else
		print_ok
	fi
}


test_3_lpath()
{
	echo " * bludiste.txt"
	echo -en "6,1\n6,2\n5,2\n5,3\n5,4\n6,4\n6,5\n6,6\n5,6\n5,7\n4,7\n4,6\n4,5\n5,5\n4,5\n4,4\n3,4\n3,3\n3,2\n4,2\n4,1\n5,1\n4,1\n4,2\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n1,4\n1,3\n1,2\n1,1\n" > $REF
	./proj3 --lpath 6 1 $M2 > $TMP
	diff $REF $TMP
	if [ ! "$?" = 0 ] ; then
		END=true
	fi

	echo " * maze_from_right.txt"
	echo -en "3,4\n3,3\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n1,4\n1,3\n1,2\n1,1\n" > $REF
	./proj3 --lpath 3 4 $M1 > $TMP
	diff $REF $TMP
	if [ ! "$?" = 0 ] ; then
		END=true
	fi

	rm -f $TMP $REF
	if [ "$END" = true ] ; then
		print_fail
	else
		print_ok
	fi
}


test_4_rpath()
{
	echo " * bludiste.txt"
	echo -en "6,1\n6,2\n5,2\n5,3\n5,4\n6,4\n6,3\n6,4\n6,5\n6,6\n5,6\n5,7\n4,7\n4,6\n4,5\n4,4\n3,4\n3,5\n3,6\n3,5\n3,4\n3,3\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n2,5\n2,6\n2,7\n3,7\n" > $REF
	./proj3 --rpath 6 1 $M2 > $TMP
	diff $REF $TMP
	if [ ! "$?" = 0 ] ; then
		END=true
	fi

	echo " * maze_from_right.txt"
	echo -en "3,4\n3,3\n3,2\n3,1\n2,1\n2,2\n2,3\n2,4\n" > $REF
	./proj3 --rpath 3 4 $M1 > $TMP
	diff $REF $TMP
	if [ ! "$?" = 0 ] ; then
		END=true
	fi

	rm -f $TMP $REF
	if [ "$END" = true ] ; then
		print_fail
	else
		print_ok
	fi
}


test_5_valgrind()
{
	RESULT=$(valgrind ./proj3 --test $M1 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --test small.txt"
		END=true
	fi

	RESULT=$(valgrind ./proj3 --test $M2 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --test bludiste.txt"
		END=true
	fi

	RESULT=$(valgrind ./proj3 --test $M4 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --test biggerMap.txt"
		END=true
	fi

	RESULT=$(valgrind ./proj3 --rpath 6 1 $M2 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --rpath CORRECT_INPUT"
		END=true
	fi

	RESULT=$(valgrind ./proj3 --lpath 6 1 $M1 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --lpath BAD_INPUT"
		END=true
	fi

	RESULT=$(valgrind ./proj3 --lpath 6 1 asdf 2>&1 > /dev/null | grep "All heap blocks were freed" | wc -l)
	if [ "$RESULT" = 0 ] ; then
		echo " ! MEMORY LEAK --lpath NOT_EXIST_FILE"
		END=true
	fi

	rm -f $TMP
	if [ "$END" = true ] ; then
		print_fail
	else
		print_ok
	fi
}

#====================================================================

echo -e "\e[1;33m#1 Print help && others\e[0m"
test_1_help

echo -e "\n\e[1;33m#2 Check maze (--test)\e[0m"
create_mazes
test_2_test

echo -e "\n\e[1;33m#3 Left path\e[0m"
test_3_lpath

echo -e "\n\e[1;33m#4 Right path\e[0m"
test_4_rpath

echo -e "\n\e[1;33m#5 Valgrind\e[0m"
test_5_valgrind

delete_mazes_exit

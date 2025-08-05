//////////////////////////////////////////////////////////////
//  file:           tail.c                                  //
//  purpose:        IJC-DU2, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-19                              //
//////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define IMPLEMENTATION_LIMIT 1024

const char *usage = "Proper usage: ./tail [-n COUNT] [filename]\n";

// checks if the COUNT in -n COUNT is a valid string
bool is_valid(char *str)
{
	// only 0-9 are valid and '+' or '-' on the first position
	for (unsigned i = 0; i < strlen(str); i++)
		if (!((i == 0 && (str[i] == '+' || str[i] == '-')) || (str[i] >= '0' && str[i] <= '9')))
			return false;
	return true;
}

// counts all lines in file
int count_lines(FILE *source) {
	int counter = 0;
	for (int c = fgetc(source); c != EOF; c = fgetc(source)) {
		if (c == '\n')
			counter++;
	}
	return counter;
}

int main(int argc, char **argv)
{
	int lines_to_print = 10; // by default
	bool plus = false;
	FILE *file_ptr = stdin; // by default

	// arguments checking
	if (argc == 2) { // only a file is given
		file_ptr = fopen(argv[1], "r");
		if (file_ptr == NULL) {
			fprintf(stderr, "ERROR 1: Could not open file %s\n", argv[1]);
			return 1;
		}
	}
	else if ((argc >= 3) && (strcmp(argv[1], "-n") == 0)) { // -n COUNT is given
		if (is_valid(argv[2])) {
			lines_to_print = atoi(argv[2]);
			plus = (argv[2][0] == '+');
			bool minus = (argv[2][0] == '-');
			if (minus) // converting to positive integer
				lines_to_print = -lines_to_print;
			if (plus && lines_to_print == 0) // tail -n +0 behaves as if tail -n +1 was inserted
				lines_to_print++;
		} 
		else { 
			fprintf(stderr, "ERROR 2: Invalid COUNT. COUNT is not a number\n%s", usage);
			return 2;
		}
		if (argc > 3) { // if an additional file is also given
			file_ptr = fopen(argv[3], "r");
			if (file_ptr == NULL) {
				fprintf(stderr, "ERROR 1: Could not open file %s\n", argv[1]);
				return 1;
			}
		}
	}

	int lines_in_file = count_lines(file_ptr);
	rewind(file_ptr);

	int start_at_line = lines_in_file - lines_to_print;
	if (plus) start_at_line = lines_to_print - 1; // so the line indexed by argument COUNT is also printed

	bool error_printed = false; // this is so that the overflow error is printed only once
	for(int i = 0; i < lines_in_file; i++) {
		int char_counter = 0;
		int c = fgetc(file_ptr);
		for (; c != '\n' && c != EOF; c = fgetc(file_ptr)) { // excludes \n and EOF
			char_counter++;
			if (char_counter >= IMPLEMENTATION_LIMIT && !error_printed) {
				fprintf(stderr, "WARNING: Exceeded character implementation limit on line %d.\n", (i+1));
				error_printed = true; // asserting, that the error message above is going to be printed only once
			}
			else if (i >= start_at_line && char_counter < IMPLEMENTATION_LIMIT) {
				putchar(c); // only printing when on correct line and not overflowing the limit
			}
		}
		if (i >= start_at_line && c != EOF) // finally print newline
			putchar('\n');
	}
	fclose(file_ptr);
	return 0;
}
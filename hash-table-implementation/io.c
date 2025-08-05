//////////////////////////////////////////////////////////////
//  file:           io.c                                    //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "io.h"

int get_word(char *s, int max, FILE *f)
{
	int c;
	// skipping whitespaces
	for (c = fgetc(f); isspace(c) && c != EOF; c = fgetc(f));

	if (c == EOF) return EOF;
	else ungetc(c, f); // returning back the non whitespace character

	// reading until whitespace or EOF or the limit has been reached
	int counter = 0; // chceking the max limit
	for (c = fgetc(f); !isspace(c) && c != EOF && counter < max; c = fgetc(f))
		s[counter++] = c;

	s[counter] = '\0'; // properly ending the string

	if (counter == max) { // when the string is too long
		while (!isspace(c) && c != EOF) { // read untli next word
			c = fgetc(f);
			counter++;
		}
		if (c == EOF) return EOF;
		else ungetc(c, f); // 
	}
	return counter;
}
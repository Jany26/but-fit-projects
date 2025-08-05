//////////////////////////////////////////////////////////////
//  file:           io.h                                    //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-22                              //
//////////////////////////////////////////////////////////////

#ifndef IO_H
#define IO_H

#include <stdio.h> // fprintf
#include <ctype.h> // isspace()

/**
 * @brief Reads a string from a file (up to max characters) until a whitespace is reached.
 * @param s Where the string gets stored.
 * @param max Limits the length of the string to max-1 characters (last one is '\0')
 * @param f From which file the word is being read.
 * @return Amount of characters read. Returns EOF when EOF is reached.
 */
int get_word(char *s, int max, FILE *f);

#endif // IO_H
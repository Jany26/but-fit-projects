//////////////////////////////////////////////////////////////
//  file:           error.h                                 //
//  purpose:        IJC-DU1, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#ifndef ERROR_H
#define ERROR_H

// variadic function that prints out a warning message into stderr
void warning_msg(const char *fmt, ...);

// variadic function that prints out an error message into stderr
// ends the program with exit code 1
void error_exit(const char *fmt, ...);

#endif
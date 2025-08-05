//////////////////////////////////////////////////////////////
//  file:           error.c                                 //
//  purpose:        IJC-DU1, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "error.h"

void warning_msg(const char *fmt, ...) 
{
	va_list arguments;
	va_start(arguments, fmt);
	fprintf(stderr, "CHYBA: ");
	vfprintf(stderr, fmt, arguments);
	fprintf(stderr, "\n");
	va_end(arguments);
}

void error_exit(const char *fmt, ...)
{
	va_list arguments;
	va_start(arguments, fmt);
	fprintf(stderr, "CHYBA: ");
	vfprintf(stderr, fmt, arguments);
	fprintf(stderr, "\n");
	va_end(arguments);
	exit(1);
}
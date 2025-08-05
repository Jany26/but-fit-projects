//////////////////////////////////////////////////////////////
//  file:           htab_private.c                          //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#ifndef HTAB_PRIVATE_H
#define HTAB_PRIVATE_H

#include "htab.h"

struct htab {
	size_t size; // number of data entries (keys = words found)
	size_t arr_size; // number of buckets
	struct htab_item *array[]; // bucket array (pointers to linked lists with entries)
};

struct htab_item {
	htab_key_t key; // word 
	htab_value_t data; // number of occurences of the word (word count)
	struct htab_item *next; // pointing to next item (entry in linked list)
};

#endif // HTAB_PRIVATE_H
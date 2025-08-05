//////////////////////////////////////////////////////////////
//  file:           htab_bucket_count.c                     //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-19                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Returns the size of the hash table (array of pointers).
 * @param t Pointer to the hash table.
 */
size_t htab_bucket_count(const htab_t * t)
{
	return t->arr_size;
}
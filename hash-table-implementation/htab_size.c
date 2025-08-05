//////////////////////////////////////////////////////////////
//  file:           htab_size.c                             //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-19                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Counts all entries in the hash table.
 * @param t Pointer to the hash table.
 * @return Number of entries in the hash table (not array size, actual data structures stored).
 */
size_t htab_size(const htab_t * t)
{
	return t->size;
}
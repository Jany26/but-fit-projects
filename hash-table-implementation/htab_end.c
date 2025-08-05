//////////////////////////////////////////////////////////////
//  file:           htab_end.c                              //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Sets the iterator to point "after" the last entry (non-existent or NULL).
 * Basically invalidates the iterator.
 * @param t Hash table structure.
 */
htab_iterator_t htab_end(const htab_t * t)
{
	htab_iterator_t tmp = {.ptr = NULL, .t = t, .idx = t->arr_size};
	return tmp;
}
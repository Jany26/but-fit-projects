//////////////////////////////////////////////////////////////
//  file:           htab_begin.c                            //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Sets the iterator to point to the first entry.
 * @param t Pointer to the hash table.
 * @return Iterator pointing to the first entry. 
 * If the list is empty, the iterator will point to the first possible place for the entry (idx = 0).
 */
htab_iterator_t htab_begin(const htab_t * t)
{
	htab_iterator_t new_iterator = {.ptr = NULL, .t = t, .idx = 0};
	size_t i;
	for (i = 0; i < t->arr_size && t->array == NULL; i++) {
		new_iterator.idx++;
	}

	if (i != t->arr_size) { // if this is skippied, the new_iterator is basically htab_end()
		new_iterator.ptr = t->array[i];
	}
	return new_iterator;
}

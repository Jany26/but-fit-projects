//////////////////////////////////////////////////////////////
//  file:           htab_erase.c                            //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Erases the entry pointed to by the iterator from the table.
 * @param t Hash table structure.
 * @param it Iterator pointing to an entry to erase.
 * @warning The iterator will become invalid after using this function.
 */
void htab_erase(htab_t *t, htab_iterator_t it)
{
	struct htab_item *tmp = t->array[it.idx];
	if (tmp == it.ptr) {
		t->array[it.idx] = it.ptr->next;
	}
	else {
		while (tmp->next != it.ptr) {
			tmp = tmp->next;
		}
		tmp->next = it.ptr->next;
	}
	free(it.ptr);
	it.idx = t->arr_size;
}
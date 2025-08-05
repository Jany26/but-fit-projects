//////////////////////////////////////////////////////////////
//  file:           htab_iterator_next.c                    //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Sets the iterator on the next entry in the hash table. (iterator++)
 * @param it Current iterator.
 * @return The iterator pointing to the next item (entry) in the table.
 */
htab_iterator_t htab_iterator_next(htab_iterator_t it)
{
	if (it.ptr->next != NULL) { // inside the linked list
		it.ptr = it.ptr->next;
		return it;
	}
	else { // at the end of the linked list
		it.idx++;
		while (it.idx < it.t->arr_size) { // find next non empty bucket
			if (it.t->array[it.idx] != NULL) {
				it.ptr = it.t->array[it.idx];
				return it;
			}
			it.idx++;
		} 
	}
	return htab_end(it.t); // did not find a non empty bucket
}
//////////////////////////////////////////////////////////////
//  file:           htab_clear.c                            //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Destroys all entries in the hash table (frees them from memory).
 * The table will still remain in memory, although empty (no entries).
 * @param t Pointer to the hash table to clear.
 */
void htab_clear(htab_t * t)
{
	struct htab_item *tmp;
	struct htab_item *helper;   // for pointer swapping
	for (size_t i = 0; i < t->arr_size; i++) {
		tmp = t->array[i];
		t->array[i] = NULL;     // disconnecting from the list
		while (tmp != NULL) {
			helper = tmp->next; // saving the pointer to next entry
			free((void *)tmp->key); // string
			free(tmp);          // struct
			tmp = helper;       // setting the pointer to next entry (tmp = tmp->next)
		}
	}
	t->size = 0; // now the hash table has no entries
}
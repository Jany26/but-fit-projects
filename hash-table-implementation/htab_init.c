//////////////////////////////////////////////////////////////
//  file:           htab_init.c                             //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Constructs the hash table.
 * @param n Size of array of the created table.
 * @return Initialized hash table or NULL, if malloc fails.
 */
htab_t *htab_init(size_t n)
{
	htab_t *table = malloc(n * sizeof(struct htab_item *) + sizeof(htab_t));
	if (table == NULL) {
		return NULL;
	}
	table->size = 0;
	table->arr_size = n;
	for (size_t i = 0; i < n; i++) {
		table->array[i] = NULL;
	}
	return table;
}
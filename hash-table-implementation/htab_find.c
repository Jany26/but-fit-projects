//////////////////////////////////////////////////////////////
//  file:           htab_find.c                             //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Finds an entry with the specified key in the table.
 * @param t Hash table structure to be parsed.
 * @param key The search is trying to find an entry with this string of characters.
 * @return If found, returns an iterator pointing at it, otherwise returns htab_end().
 * @see htab_end()
 */
htab_iterator_t htab_find(htab_t * t, htab_key_t key)
{
	size_t index = htab_hash_fun(key) % t->arr_size;
	htab_iterator_t tmp = {.ptr = t->array[index], .t = t, .idx = index};
	while (tmp.ptr != NULL) {
		if (strcmp(tmp.ptr->key, key) == 0)
			return tmp;
		tmp.ptr = tmp.ptr->next;
	}
	return htab_end(t);
}
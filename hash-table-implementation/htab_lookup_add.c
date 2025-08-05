//////////////////////////////////////////////////////////////
//  file:           htab_lookup_add.c                       //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Finds the entry with a certain key in the hash table. If not found, the entry is added.
 * @param t Which hash table to parse.
 * @param key String to be found (or added).
 * @return The iterator pointing to the found (or added) entry in the table.
 * If the allocation is unsuccessful, htab_end() is returned.
 * @see htab_end(), htab_find()
 */
htab_iterator_t htab_lookup_add(htab_t * t, htab_key_t key)
{
	htab_iterator_t tmp = htab_find(t, key);
	// case 1 - already in the table
	if (tmp.ptr != NULL) {
		htab_iterator_set_value(tmp, htab_iterator_get_value(tmp) + 1); // data++
		return tmp;
	}
	// case 2 - not in the table => find the right bucket
	size_t index = htab_hash_fun(key) % t->arr_size;
	tmp.ptr = t->array[index];
	tmp.idx = index;
	// create entry
	struct htab_item *new_entry = (struct htab_item *) malloc(sizeof(struct htab_item));
	if (new_entry == NULL) {
		return htab_end(t);
	}
	new_entry->next = NULL;
	new_entry->data = 1;
	new_entry->key = (htab_key_t) malloc(strlen(key) + 1);
	if (new_entry->key == NULL) {
		free(new_entry);
		return htab_end(t);
	}
	strcpy((char *)new_entry->key, key);
	// case 2a - add the entry to an empty bucket
	if (tmp.ptr == NULL) { 
		tmp.ptr = new_entry;
		t->array[index] = new_entry; // update bucket pointer
		t->size++;
		return tmp;
	}
	// case 2b - add to the end of non empty bucket
	while (tmp.ptr->next != NULL) {
		tmp.ptr = tmp.ptr->next;
	}
	tmp.ptr->next = new_entry;
	tmp.ptr = new_entry;
	t->size++;
	return tmp;
}

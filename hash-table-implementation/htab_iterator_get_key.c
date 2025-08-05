//////////////////////////////////////////////////////////////
//  file:           htab_iterator_get_key.c                 //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Retrieves the key (char *) from the entry the iterator points to
 * @param it Iterator pointing to the entry.
 */
htab_key_t htab_iterator_get_key(htab_iterator_t it) 
{
	return it.ptr->key;
}
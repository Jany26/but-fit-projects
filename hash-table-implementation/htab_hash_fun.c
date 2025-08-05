//////////////////////////////////////////////////////////////
//  file:           htab_hash_fun.c                         //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Hashing function - calculates a value (hash code) based on key (character array).
 * @param t str Key.
 * @return Hash code calculated using sdbm based on key string.
 */
size_t htab_hash_fun(htab_key_t str)
{
	size_t h = 0;
	const unsigned char *p;
	for(p = (const unsigned char *) str; *p != '\0'; p++)
		h = 65599 * h + *p;
	return h;
}
//////////////////////////////////////////////////////////////
//  file:           htab_free.c                             //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h"

/**
 * @brief Destructor of the hash table. Calls htab_clear().
 * @param t Hash table structure to be destroyed. (freed from memory)
 * @see htab_clear().
 */
void htab_free(htab_t * t)
{
	htab_clear(t);
	//free(t->array); if calloc was used in htav_init
	free(t);
}
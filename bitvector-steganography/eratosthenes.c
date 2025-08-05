//////////////////////////////////////////////////////////////
//  file:           eratosthenes.c                          //
//  purpose:        IJC-DU1, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "eratosthenes.h"

void eratosthenes(bitset_t array_name)
{
	bitset_setbit(array_name, 0, 1);
	bitset_setbit(array_name, 1, 1);
	
	for (bitset_index_t i = 2; i < sqrt(bitset_size(array_name)); i++) {
		if (bitset_getbit(array_name, i) == 0) {
			size_t j;
			for (j = 2*i; j < bitset_size(array_name); j += i) {
				bitset_setbit(array_name, j, 1);
			}
		}
	}
	return;
}
//////////////////////////////////////////////////////////////
//  file:           bitset.h                                //
//  purpose:        IJC-DU1, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include <assert.h>
#include <limits.h>
#include "error.h"


#ifndef BITSET_H
#define BITSET_H

typedef unsigned long bitset_index_t;
typedef bitset_index_t *bitset_t;

// number of bits in the array element (unsigned long = default 64)
#define ELEMENT_BITSIZE (sizeof(bitset_index_t) * CHAR_BIT)

// computes, in which element of the unsigned long arraz the bit index is located
// takes into account, that element with index 0 is reserved for size of array in bits
#define element_count(bit_size) (bit_size / ELEMENT_BITSIZE + (bit_size % ELEMENT_BITSIZE ? 2 : 1))

// computes array index depending on the bit index
// default (unsigned long = 8 bytes): [0] = size; [1] = bits with indexes <0,63>; [2] = bits with indexes <64,127> ...
#define element_index(bit_index) (bit_index / ELEMENT_BITSIZE + 1)

// creates a local array (on the stack) and initializes it (first element = bitsize, other = 0)
#define bitset_create(bitset_name, bit_size) \
	_Static_assert(bit_size > 0, "bitset_create : Velikost pole musi byt vetsi nez nula."); \
	bitset_index_t bitset_name[element_count(bit_size)] = {(bitset_index_t) bit_size}
	
// creates a dynamic array (on the heap) - equivalent of the bitset_create
#define bitset_alloc(bitset_name, bit_size) \
	assert(bit_size > 0); \
	bitset_t bitset_name = calloc(element_count(bit_size), sizeof(bitset_index_t)); \
	if (bitset_name == NULL) \
		error_exit("bitset_alloc: Chyba alokace pameti"); \
	*bitset_name = (bitset_index_t) bit_size

#ifndef USE_INLINE

	// frees the dynamic array
	#define bitset_free(bitset_name) free(bitset_name)

	// returns the bitsize of the array
	#define bitset_size(bitset_name) bitset_name[0]

	// sets the bit on index to a value of expression (non-zero = 1, zero = 0)
	#define bitset_setbit(bitset_name, index, expression) \
		if ((bitset_index_t) index >= bitset_size(bitset_name)) \
		(error_exit("bitset_setbit: Index %lu mimo rozsah 0..%lu", (bitset_index_t) index, bitset_size(bitset_name) - 1)); \
			(expression) ? \
				(bitset_name[element_index(index)] |= (1UL << (index % ELEMENT_BITSIZE))) : \
				(bitset_name[element_index(index)] &= ~(1UL << (index % ELEMENT_BITSIZE)))

	// returns bit on the index = nonzero value = 1, zero value = 0
	#define bitset_getbit(bitset_name, index) \
		((bitset_index_t) index >= bitset_size(bitset_name)) ? \
		(error_exit("bitset_setbit: Index %lu mimo rozsah 0..%lu", (bitset_index_t) index, bitset_size(bitset_name) - 1), 1) : \
		((bitset_name[element_index(index)] & (1UL << (index % ELEMENT_BITSIZE))) != 0)

// inline functions do the same as their macro equivalents
// they are defined if the program is compiled with the -DUSE_INLINE argument
#else
bitset_setbit(bitset_name, index, expression) \
	inline void bitset_free(bitset_t bitset_name)
	{
		free(bitset_name);
		return;
	}

	inline bitset_index_t bitset_size(bitset_t bitset_name)
	{
		return bitset_name[0];
	}

	inline void bitset_setbit(bitset_t bitset_name, const bitset_index_t index, const int expression)
	{
		if (index >= bitset_size(bitset_name)) {
			(error_exit("bitset_setbit: Index %lu mimo rozsah 0..%lu", 
				 index, bitset_size(bitset_name) - 1));
		}
		if (expression) {
			(bitset_name[element_index(index)]) |=  (1UL << (index % ELEMENT_BITSIZE));
		}
		else {
			(bitset_name[element_index(index)]) &= ~(1UL << (index % ELEMENT_BITSIZE));
		}
		return;
	}

	inline int bitset_getbit(bitset_t bitset_name, const bitset_index_t index)
	{
		if (index >= bitset_size(bitset_name)) {
			(error_exit("bitset_setbit: Index %lu mimo rozsah 0..%lu", 
				 index, bitset_size(bitset_name) - 1));
		}
		return ((bitset_name[element_index(index)] & (1UL << (index % ELEMENT_BITSIZE))) != 0);
	}

#endif /* USE_INLINE */

#endif /* BITSET_H */
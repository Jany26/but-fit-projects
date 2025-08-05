//////////////////////////////////////////////////////////////
//  file:           bitset.c                                //
//  purpose:        IJC-DU1, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "bitset.h"

#ifdef USE_INLINE

extern inline void bitset_free(bitset_t bitset_name);
extern inline bitset_index_t bitset_size(bitset_t bitset_name);
extern inline void bitset_setbit(bitset_t bitset_name, const unsigned long index, const int expression);    
extern inline int bitset_getbit(bitset_t bitset_name, const unsigned long index);

#endif 
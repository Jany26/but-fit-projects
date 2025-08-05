//////////////////////////////////////////////////////////////
//  file:           eratosthenes.h                          //
//  purpose:        IJC-DU1, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#ifndef ERATOSTHENES_H
#define ERATOSTHENES_H

#include <math.h>
#include "bitset.h"

// sets bit values of indexes in bitset arr depending on eratosthenes sieve
// 0 - index of this bit in bitset is a prime
// 1 - index of this bit in bitset is not a prime
// first couple of bits should look like this :110010101110...
void eratosthenes(bitset_t arr);

#endif
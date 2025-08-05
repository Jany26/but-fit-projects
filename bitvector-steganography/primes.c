//////////////////////////////////////////////////////////////
//  file:           primes.c                                //
//  purpose:        IJC-DU1, task a)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "eratosthenes.h"
#include <time.h>

#define SIZE 500000000

int main(void)
{
	clock_t start = clock();

	bitset_create(eratosthenes_sieve, SIZE);
	eratosthenes(eratosthenes_sieve);
	bitset_index_t last_primes[10];
	for (int i = SIZE - 1, j = 10; i > 1 && j != 0; i--)
		if (bitset_getbit(eratosthenes_sieve, i) == 0)
			last_primes[--j] = i;

	for (int j = 0; j < 10; j++)
		printf("%ld\n", last_primes[j]);
	
	fprintf(stderr, "Time=%.3g\n", (double)(clock() - start)/CLOCKS_PER_SEC);
	return 0;
}
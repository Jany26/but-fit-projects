//////////////////////////////////////////////////////////////
//  file:           wordcount.c                             //
//  purpose:        IJC-DU2, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-04-23                              //
//////////////////////////////////////////////////////////////

#include "htab_private.h" // also includes htab.h
#include "io.h" // get_word

extern inline bool htab_iterator_valid(htab_iterator_t it);
extern inline bool htab_iterator_equal(htab_iterator_t it1, htab_iterator_t it2);

// It is good practice to make the size of hash table close to powers of 2. 
// Size should be a prime number to minimize the number of collisions when 
// using mod to determine index. 
// I chose 2^16. Closest prime smaller than 65536 is 65521.
// Hash table should be 1.3x the amount of stored entries. 
// So in order to keep the load factor below 75%, the optimal amount of data
// entries would be 49140.
#define HTAB_SIZE 65521
#define MAX_CHAR_LIMIT 127

int main() 
{
	char *word = malloc(MAX_CHAR_LIMIT);
	htab_t *table = htab_init(HTAB_SIZE);
	htab_iterator_t iterator;
	bool error_long_word = false;

	// loading into the hash table
	for (int i = get_word(word, MAX_CHAR_LIMIT, stdin); i != EOF; i = get_word(word, MAX_CHAR_LIMIT, stdin)) {
		iterator = htab_lookup_add(table, word);
		if (i >= MAX_CHAR_LIMIT && !error_long_word) {
			fprintf(stderr,"WARNING: Found a word too long. End will be truncated.\n");
			error_long_word = true; // WARNING will be printed max once
		}
	}
	free(word);
	// 
	for (iterator = htab_begin(table); htab_iterator_valid(iterator); iterator = htab_iterator_next(iterator)) {
		printf("%s\t", htab_iterator_get_key(iterator));
		printf("%d\n", htab_iterator_get_value(iterator));
	}

	htab_free(table);
	return 0;
}
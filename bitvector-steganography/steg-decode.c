//////////////////////////////////////////////////////////////
//  file:           steg-decode.c                           //
//  purpose:        IJC-DU1, task b)                        //
//  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     //
//  compiled:       gcc 7.5.0 - Ubuntu 18.04                //
//  last update:    2020-03-18                              //
//////////////////////////////////////////////////////////////

#include "ppm.h"
#include "eratosthenes.h"

#define char_setbit(character, bit_index, value) \
	(value) ? (character |= (1 << bit_index)) \
			: (character &= ~(1 << bit_index))

int main(int argc, char **argv)
{
	if (argc < 2)
		error_exit("Chybi jmeno souboru");

	struct ppm *picture = ppm_read(argv[1]);
	if (picture == NULL) {
		error_exit("pri cteni souboru (chybny format)");
	}

	bitset_alloc(eratosthenes_sieve, 3 * picture->xsize * picture->ysize);
	eratosthenes(eratosthenes_sieve);
	
	unsigned char decoded_char = ' ';

	// main data decryption algorithm
	int bit_counter = 0;
	// the encrypted bits are stored in LSB of the prime indexes of the picture->data starting from 23
	for (unsigned long i = 23; i < bitset_size(eratosthenes_sieve); i++) {
		if (bitset_getbit(eratosthenes_sieve, i) == 0) {
			// storing decrypted bits from LSB to MSB
			char_setbit(decoded_char, (bit_counter % CHAR_BIT), (unsigned char) picture->data[i] & 1);
			bit_counter++;
			// every 8th time a bit is stored in the decoded_char (so it is fully loaded),
			// decoded_char is printed on stdout and the program checks for '\0'
			if ((bit_counter % CHAR_BIT) == 0) {
				putchar(decoded_char);
				if (decoded_char == '\0') 
					break;
			}
		}
	}
	
	// the encoded message string is not properly finished
	if (decoded_char != '\0') {
		printf("\n");
		bitset_free(eratosthenes_sieve);
		ppm_free(picture);		
		error_exit("Sprava neni ukoncena '\\0'");
	}
	
	// the decryption and printing was successful
	printf("\n");
	bitset_free(eratosthenes_sieve);
	ppm_free(picture);
	return 0;
}
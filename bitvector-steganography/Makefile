##############################################################
##  file:           Makefile                                ##
##  purpose:        IJC-DU1                                 ##
##  author:         Ján Maťufka, xmatuf00, 1BIT FIT VUT     ##
##  compiled:       Ubuntu 18.04                            ##
##  last update:    2020-03-18                              ##
##############################################################

CFLAGS = gcc -O2 -g -std=c11 -pedantic -Wall -Wextra
LDFLAGS = -lm 

all: primes primes-i steg-decode

run: primes primes-i steg-decode
	ulimit -s unlimited && ./primes
	ulimit -s unlimited && ./primes-i

##############################################################
##                     TASK a) - PRIMES                     ##
##############################################################

## USING MACROS

primes.o: Makefile primes.c
	$(CFLAGS) -c primes.c

eratosthenes.o: Makefile eratosthenes.h eratosthenes.c
	$(CFLAGS) -c eratosthenes.c

bitset.o: Makefile bitset.c bitset.h
	$(CFLAGS) -c bitset.c

## USING INLINE FUNCTIONS

primes-i.o: Makefile primes.c
	$(CFLAGS) -c -DUSE_INLINE primes.c -o primes-i.o

eratosthenes-i.o: Makefile eratosthenes.h eratosthenes.c
	$(CFLAGS) -c -DUSE_INLINE eratosthenes.c -o eratosthenes-i.o 

bitset-i.o: Makefile bitset.c bitset.h
	$(CFLAGS) -c -DUSE_INLINE bitset.c -o bitset-i.o

##############################################################
##                   TASK b) - STEG DECODE                  ##
##############################################################

steg-decode.o: Makefile steg-decode.c
	$(CFLAGS) -c steg-decode.c

ppm.o: Makefile ppm.c ppm.h
	$(CFLAGS) -c ppm.c

error.o: Makefile error.c error.h
	$(CFLAGS) -c error.c

##############################################################
##                        EXECUTABLES                       ##
##############################################################

primes:	error.o bitset.o eratosthenes.o primes.o
	$(CFLAGS) primes.o error.o bitset.o eratosthenes.o -o primes $(LDFLAGS)

primes-i: error.o bitset-i.o eratosthenes-i.o primes-i.o
	$(CFLAGS) primes-i.o error.o bitset-i.o eratosthenes-i.o -o primes-i $(LDFLAGS)

steg-decode: error.o bitset.o eratosthenes.o ppm.o steg-decode.o
	$(CFLAGS) steg-decode.o error.o bitset.o eratosthenes.o ppm.o -o steg-decode $(LDFLAGS)

##############################################################
##                           EXTRA                          ##
##############################################################

clean:
	rm *.o primes primes-i steg-decode

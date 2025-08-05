##############################################################
##  file:           Makefile                                ##
##  purpose:        IJC-DU2                                 ##
##  author:         Ján Maťufka, xmatuf00. 1BIT FIT VUT     ##
##  compiled on:    Ubuntu 18.04, gcc 7.5.0                 ##
##  last update:    2020-04-23                              ##
##############################################################

CC=gcc
CFLAGS=-std=c99 -Wall -Wextra -pedantic -g

##############################################################
##                       EXECUTABLES                        ##
##############################################################

all: tail wordcount wordcount-dynamic

tail: tail.o
	$(CC) $(CFLAGS) $^ -o $@

wordcount: wordcount.o io.o libhtab.a
	$(CC) $(CFLAGS) $^ -o $@

wordcount-dynamic: wordcount.o io.o libhtab.so
	$(CC) -Wl,-rpath,. $(CFLAGS) -fPIC $^ -o $@

##############################################################
##                      TASK a) - TAIL                      ##
##############################################################

tail.o: tail.c
	$(CC) $(CFLAGS) $^ -c

##############################################################
##                   TASK b) - WORD COUNT                   ##
##############################################################

io.o: io.c io.h
	$(CC) $(CFLAGS) $^ -c

wordcount.o: wordcount.c
	$(CC) $(CFLAGS) $^ -c

##############################################################
##               TASK b) - HASH TABLE LIBRARY               ##
##############################################################

HTAB_DEPENDENCIES := htab_begin.o htab_bucket_count.o htab_clear.o
HTAB_DEPENDENCIES += htab_end.o htab_erase.o htab_find.o htab_free.o htab_hash_fun.o htab_init.o
HTAB_DEPENDENCIES += htab_iterator_get_key.o htab_iterator_get_value.o htab_iterator_next.o
HTAB_DEPENDENCIES += htab_iterator_set_value.o htab_lookup_add.o htab_size.o

libhtab.a: $(HTAB_DEPENDENCIES)
	ar rcs $@ $^

libhtab.so: $(HTAB_DEPENDENCIES)
	$(CC) -Wl,-rpath,. $(CFLAGS) -shared -fPIC $(HTAB_DEPENDENCIES) -o $@

##############################################################
##              TASK b) - HASH TABLE FUNCTIONS              ##
##############################################################

htab_begin.o: htab.h htab_private.h htab_begin.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_bucket_count.o: htab.h htab_private.h htab_bucket_count.c 
	$(CC) $(CFLAGS) -fPIC -c $^
htab_clear.o: htab.h htab_private.h htab_clear.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_end.o: htab.h htab_private.h htab_end.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_erase.o: htab.h htab_private.h htab_erase.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_find.o: htab.h htab_private.h htab_find.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_free.o: htab.h htab_private.h htab_free.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_hash_fun.o: htab.h htab_private.h htab_hash_fun.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_init.o: htab.h htab_private.h htab_init.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_iterator_get_key.o: htab.h htab_private.h htab_iterator_get_key.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_iterator_get_value.o: htab.h htab_private.h htab_iterator_get_value.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_iterator_next.o: htab.h htab_private.h htab_iterator_next.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_iterator_set_value.o: htab.h htab_private.h htab_iterator_set_value.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_lookup_add.o: htab.h htab_private.h htab_lookup_add.c
	$(CC) $(CFLAGS) -fPIC -c $^
htab_size.o: htab.h htab_private.h htab_size.c
	$(CC) $(CFLAGS) -fPIC -c $^

##############################################################
##                        MAINTENANCE                       ##
##############################################################

zip:
	zip xmatuf00 *.c *.h Makefile

clean:
	rm *.o *.so *.a tail wordcount wordcount-dynamic

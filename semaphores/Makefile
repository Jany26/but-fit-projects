#########################################################################################
# file: Makefile                                                                        #
# author: Ján Maťufka - xmatuf00@stud.fit.vutbr.cz                                      #
#                                                                                       #
# Makefile for 2nd project for IOS (Operating Systems) Course - 1BIT FIT VUT - 2020     #
# Solution to the modified Faneuil Hall problem - process synchronization               #
#                                                                                       #
# date: 2020-05-05                                                                      #
#########################################################################################

CC = gcc
CFLAGS = -std=gnu99 -Wall -Wextra -Werror -pedantic
LDFLAGS = -pthread

proj2: proj2.o definitions.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

proj2.o: proj2.c definitions.c definitions.h
	$(CC) $(CFLAGS) $< -c

definitions.o: definitions.c definitions.h
	$(CC) $(CFLAGS) $^ -c

clean:
	rm -f *.o *.gch proj2 proj2.out

zip:
	zip proj2 *.c *.h Makefile
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -pedantic
LFLAGS = -lelf
all:
	$(CC) $(CFLAGS) analyze-elf.c -o analyze-elf $(LFLAGS)

clean:
	rm analyze-elf

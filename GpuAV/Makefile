CC = gcc
CFLAGS = -std=c99

run:	main.c parse.c
	$(CC) $(CFLAGS) $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial *.o

CC = gcc
CFLAGS = -std=c99

serial:	serial.c
	$(CC) $(CFLAGS) $^ -o $@

.c.o:
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f serial *.o

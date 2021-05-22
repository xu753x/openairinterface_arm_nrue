CC=gcc
CFLAGS=-Wall -g -I.

OBJS=bench.o nr_pdcp_entity.o nr_pdcp_integrity_nia2.o nr_pdcp_security_nea2.o \
     nr_pdcp_sdu.o

bench: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lcrypto -lnettle

%.o: ../%.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f bench $(OBJS)

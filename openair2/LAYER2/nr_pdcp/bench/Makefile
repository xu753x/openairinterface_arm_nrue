CC=gcc
CFLAGS=-Wall -g -I.

OBJS=bench.o nr_pdcp_entity.o nr_pdcp_integrity_nia2.o nr_pdcp_security_nea2.o \
     nr_pdcp_sdu.o

OBJS_optimized=bench.o nr_pdcp_entity.o nr_pdcp_integrity_nia2_optimized.o \
     nr_pdcp_security_nea2_optimized.o nr_pdcp_sdu.o

all: bench bench_optimized

bench: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lcrypto -lnettle

bench_optimized: $(OBJS_optimized)
	$(CC) $(CFLAGS) -o $@ $(OBJS_optimized) -lcrypto -lnettle

%.o: ../%.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f bench $(OBJS) bench_optimized $(OBJS_optimized)

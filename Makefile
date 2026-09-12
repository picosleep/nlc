CC ?= gcc
BASE_CFLAGS := -O3 -std=gnu11 -Wall -Wextra -Wshadow -Wconversion -Wformat=2 -Wstrict-prototypes
CFLAGS ?= $(BASE_CFLAGS)
GIT_COMMIT := $(shell git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
RECORDED_FLAGS := $(CFLAGS) -pthread -lm
CPPFLAGS += -DCXL_BENCH_BUILD_FLAGS='"$(RECORDED_FLAGS)"' -DCXL_BENCH_GIT_COMMIT='"$(GIT_COMMIT)"'
LDLIBS ?= -pthread -lm

all: mbs nlc

nlc: nlc.c
	$(CC) -O2 -march=native $< -o $@ -lnuma

mbs: mbs.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f mbs nlc


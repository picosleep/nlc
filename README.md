# nlc

1. compile (with -lnuma flag)

`gcc -O2 -march=native nlc.c -o nlc -lnuma`

2. run (e.g. targeting CXL Memory-Only Node 1 )

`taskset -c 0 ./nlc 1`

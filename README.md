# nlc

1. コンパイル (-lnuma フラグの指定)

`gcc -O2 -march=native nlc.c -o nlc -lnuma`

2. 実行 (例: CXL Memory-Only Node 1 に対して実行する場合)

`taskset -c 0 ./nlc 1`

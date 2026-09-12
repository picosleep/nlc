# CXL Physical Benchmark Engine v1.1

`v1.1` is a measurement-principle reset. **Do not use v1.0 results as a baseline.**

The benchmark measures the end-to-end path from pinned CPU threads to memory pages verified on the selected NUMA node. It does **not** automatically claim that the result is the intrinsic DRAM-array limit of one CXL device. If the target cannot be conservatively identified as a single-device, one-way region candidate, the report scope remains aggregate/end-to-end.

## Build

```bash
make -f Makefile.cxl_bench_v1.1
```

Equivalent minimal build:

```bash
gcc -O3 -std=gnu11 -Wall -Wextra -Wshadow -Wconversion \
    -Wformat=2 -Wstrict-prototypes -pthread \
    cxl_bench_suite_v1.1.c -o cxl_bench_suite -lm
```

The supplied Makefile also records compiler flags and a Git commit string in the report. The program records the compiler version and, when `sha256sum` is available, the running binary SHA-256.

No `libnuma` development package is required. NUMA binding and residency queries use Linux system calls directly; CPU/node/topology data comes from sysfs.

## Typical strict run

```bash
./cxl_bench_suite 0 2 32 \
  --page-mode hugetlb \
  --huge-size 2M \
  --total-gib 8 \
  --latency-gib 1 \
  --warmup-sec 3 \
  --duration-sec 10 \
  --repeats 5 \
  --output ./cxl-bench-results
```

The default page mode is `hugetlb` and there is no silent fallback. Reserve HugeTLB pages on the target node before running. `thp` and `base` are explicit alternatives; strict THP mode rejects a mapping that was not fully backed by anonymous huge pages.

## v1.1 measurement corrections

1. **Compiler-DCE prevention** — read accumulators and the final pointer-chase value are made externally observable. `-O3` can no longer reduce the tests to counter-only loops.
2. **Single dependent latency chain** — latency is one randomized pointer chain. It is reported as nanoseconds per dependent hop. The old four-independent-chain behavior was MLP throughput, not load-to-use latency.
3. **Full 64-byte kernels** — every counted read or write operation covers a complete cache line. x86-64 NT write uses four 128-bit streaming stores; AArch64 uses four STNP pairs as a non-temporal hint. The AArch64 hint must not be interpreted as an architectural guarantee of x86-like cache-bypass behavior.
4. **Atomic start and measured time** — C11 atomics plus start/end barriers establish the measurement interval. `CLOCK_MONOTONIC_RAW` is used and measured elapsed time, not the requested sleep duration, is used for bandwidth.
5. **Topology-aware CPU affinity** — CPUs are selected from the requested CPU NUMA node intersected with the process cpuset. One CPU is reserved for latency. One logical CPU per physical core is used before SMT siblings. Every worker verifies its start/end CPU.
6. **Strict NUMA residency verification** — the mapping is bound with `MPOL_BIND | MPOL_MF_STRICT`, prefaulted, then sampled with `move_pages(..., nodes=NULL)` before and after the suite. Failure to verify the pre-run mapping is fatal; a failed post-run check invalidates all summary samples.
7. **CXL topology resolution** — the report enumerates visible CXL memdevs and regions, records region interleave settings/targets, and correlates the selected NUMA node with CXL region HPA ranges using online memory-block physical ranges. It deliberately avoids claiming a one-device limit when the topology cannot prove it.
8. **Canonical units** — raw bandwidth is `bytes/s`; display helpers include decimal GB/s and binary GiB/s. Latency is nanoseconds, not TSC/CNTVCT “cycles”.
9. **Warm-up, repetitions, statistics** — every recorded point has a warm-up phase. Repeated valid runs are summarized by median/min/max and coefficient of variation.
10. **Environment/schema/quality gates** — JSON has an independent schema version, environment/build/topology/page mapping data, raw runs, validity reasons, and summaries. Reports are written to a temporary file, `fsync()`ed, then atomically renamed.

## Traffic accounting

The bandwidth reported by the software generator is **logical full-cache-line traffic explicitly loaded/stored by the benchmark kernel**.

For ordinary write (`R2_W1`), the report does not invent RFO or eviction/writeback byte counts. Those are physical cache/coherency/fabric transactions and should be validated with available platform PMUs. v1.1 detects the presence of Linux CXL CPMU devices but does not yet program PMU counters itself.

Patterns:

| Name | Kernel traffic |
|---|---|
| `R` | 100% 64 B reads |
| `NTW` | 100% 64 B NT writes / NT hints |
| `R2_NTW1` | 2 read cache lines : 1 NT-write cache line |
| `R1_NTW1` | 1 read cache line : 1 NT-write cache line |
| `R2_W1` | 2 read cache lines : 1 ordinary 64 B write cache line |

## Result validity

A recorded run is rejected from the summary if CPU affinity failed, measured duration exceeded the tolerance, pre-run residency was not fully verified, a major page fault occurred during measurement, or post-suite residency was not fully verified.

Minor faults, context switches, NUMA balancing status, page mode and other environmental conditions remain in the raw report even when they do not independently invalidate a sample. This is intentional: operators need the evidence to decide whether a run should be repeated under a cleaner host configuration.

## Working-set rule

The aggregate bandwidth working set is constant as thread count increases; it is divided by the **active thread count**, not by `MAX_THREADS`. Preflight rejects a configured bandwidth footprint smaller than four times the detected LLC capacity when LLC size is available.

The latency working set is separate from the bandwidth working set, preventing bandwidth writers from corrupting the pointer chain.

## Output

Each invocation creates a unique directory such as:

```text
cxl-bench-results/
  20260912T012345Z-host-n2-pid1234/
    report.json
```

Important report fields include:

- tool/schema version and `v1_0_baseline_compatible: false`
- measurement scope
- compiler/build flags/Git commit/binary SHA-256
- kernel, BIOS, microcode, command line, NUMA balancing and CPU governor
- selected latency/BW CPUs
- requested and observed page mode information
- pre/post NUMA residency evidence
- CXL memdev and region topology, interleave settings and HPA overlap
- raw per-run byte counts, elapsed time, faults/context switches and validity reason
- median/min/max/CV for valid repetitions

## Operational cautions

- Run on an otherwise quiet host. Pin or move unrelated IRQ-heavy workloads away from the selected CPUs when possible.
- Keep CPU frequency/power policy stable between comparison runs; the report records the governor but does not change host policy.
- For strict comparisons, disable automatic NUMA balancing. v1.1 warns when it is enabled and still verifies page placement before/after the suite.
- Reserve enough HugeTLB pages **on the target CXL NUMA node**. A global huge-page reservation that lands on another node is rejected by residency verification.
- A NUMA node may represent an interleaved CXL region with multiple memdevs. In that case, quote the result as region/path aggregate bandwidth rather than one-device bandwidth.
- CXL CPMU availability is recorded. Physical CXL-link transaction accounting remains a separate observer function and should be added before claiming protocol-level byte efficiency from `R2_W1` or other coherency-sensitive patterns.

## Validation performed on the supplied source

The delivered source was built warning-free with:

- GCC 14.2, `-O3`, strict warning set
- Clang 17, `-O3`, same warning set

On x86-64, `objdump` inspection confirmed that the `-O3` binary retains the dependent pointer loads, full-line read memory operations, and four 16-byte `movntdq` stores per NT-written cache line. Actual CXL measurements still require validation on the target CXL host; this development environment does not expose the user's CXL hardware.

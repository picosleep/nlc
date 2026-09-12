/*
 * mbs: Memory Benchmark Suite (FKA cxl_bench_suite_v1.1.c - CXL Physical Benchmark Engine v1.1)
 *
 * Goal:
 *   Strict, reproducible CPU->CXL-memory end-to-end benchmarking with
 *   explicit validity gates. v1.0 results MUST NOT be used as a baseline.
 *
 * Build:
 *   gcc -O3 -std=gnu11 -Wall -Wextra -Wshadow -Wconversion \
 *       -pthread cxl_bench_suite_v1.1.c -o cxl_bench_suite
 *
 * Linux only. Supported ISA backends: x86-64 (SSE2 NT stores), AArch64
 * (AdvSIMD loads + STNP non-temporal-hint stores).
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdarg.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <linux/mempolicy.h>

#if defined(__x86_64__)
# include <immintrin.h>
# include <x86intrin.h>
#elif defined(__aarch64__)
# include <arm_neon.h>
#else
# error "Unsupported architecture: v1.1 supports x86-64 and AArch64."
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

#define VERSION "1.1.0"
#define SCHEMA_VERSION "1.0"
#ifndef CXL_BENCH_BUILD_FLAGS
#define CXL_BENCH_BUILD_FLAGS "not_recorded (define CXL_BENCH_BUILD_FLAGS at compile time)"
#endif
#ifndef CXL_BENCH_GIT_COMMIT
#define CXL_BENCH_GIT_COMMIT "unknown"
#endif
#define CACHE_LINE_SIZE 64ULL
#define KIB (1024ULL)
#define MIB (1024ULL * KIB)
#define GIB (1024ULL * MIB)
#define MAX_THREADS 256
#define MAX_CPUS 4096
#define MAX_CXL_MEMDEVS 64
#define MAX_CXL_REGIONS 64
#define MAX_REGION_TARGETS 64
#define MAX_PATH 4096
#define DEFAULT_DURATION_SEC 10.0
#define DEFAULT_WARMUP_SEC 3.0
#define DEFAULT_REPEATS 5
#define DEFAULT_TOTAL_GIB 4.0
#define DEFAULT_LAT_GIB 1.0
#define DEFAULT_RESIDENCY_SAMPLES 4096
#define BW_TIME_CHECK_LINES 4096ULL       /* 256 KiB between time checks */
#define LAT_TIME_CHECK_HOPS 4096ULL

/* ---------- Traffic patterns ---------- */
typedef enum {
    TP_R = 0,          /* 100% full-cache-line reads */
    TP_NTW = 1,        /* 100% full-cache-line non-temporal writes/hints */
    TP_R2_NTW1 = 2,    /* 2 read lines : 1 NT-write line */
    TP_R1_NTW1 = 3,    /* 1 read line : 1 NT-write line */
    TP_R2_W1 = 4,      /* 2 read lines : 1 ordinary write line (RFO-capable) */
    NUM_PATTERNS
} traffic_pattern_t;

static const char *tp_names[NUM_PATTERNS] = {
    "R", "NTW", "R2_NTW1", "R1_NTW1", "R2_W1"
};

/* ---------- Page mode ---------- */
typedef enum {
    PAGE_HUGETLB = 0,
    PAGE_THP = 1,
    PAGE_BASE = 2
} page_mode_t;

/* ---------- Core data ---------- */
struct node {
    struct node *next;
    uint8_t pad[CACHE_LINE_SIZE - sizeof(struct node *)];
} __attribute__((aligned(CACHE_LINE_SIZE)));

struct cpu_choice {
    int cpu;
    int package_id;
    int core_id;
    bool primary_thread;
};

struct cxl_memdev_info {
    char name[64];
    int numa_node;
    char serial[128];
    char firmware_version[128];
    char ram[64];
    char pmem[64];
    char sysfs_path[PATH_MAX];
    char canonical_path[PATH_MAX];
};

struct cxl_region_info {
    char name[64];
    char mode[64];
    char resource[64];
    char size[64];
    char interleave_ways[64];
    char interleave_granularity[64];
    int target_count;
    char targets[MAX_REGION_TARGETS][128];
    bool overlaps_target_node;
    int overlapping_memory_blocks;
};

struct environment {
    char timestamp[64];
    char hostname[256];
    char kernel_release[256];
    char architecture[128];
    char cpu_model[256];
    char cpu_microcode[64];
    char bios_version[128];
    char bios_date[64];
    char kernel_cmdline[2048];
    char compiler[256];
    char build_flags[512];
    char git_commit[128];
    char binary_sha256[128];
    char distro[256];
    char isa_backend[128];
    int numa_balancing;
    char thp_status[128];
    int cpu_node;
    int cxl_node;
    int numa_distance;
    uint64_t cxl_node_mem_total_bytes;
    uint64_t llc_bytes;
    char cpu_governor[64];
    char cpu_cpulist[4096];
    char mems_allowed_list[1024];
    int cxl_memdev_count;
    struct cxl_memdev_info memdevs[MAX_CXL_MEMDEVS];
    int cxl_region_count;
    struct cxl_region_info regions[MAX_CXL_REGIONS];
    bool cxl_sysfs_present;
    bool cxl_node_has_memdev;
    bool cxl_node_overlaps_region;
    bool cxl_identity_proven_single_device;
    char measurement_scope[128];
    bool cxl_pmu_present;
};

struct options {
    int cpu_node;
    int cxl_node;
    int max_threads;
    double total_gib;
    double latency_gib;
    double duration_sec;
    double warmup_sec;
    int repeats;
    page_mode_t page_mode;
    int huge_shift; /* 0 = kernel default HugeTLB size */
    uint64_t seed;
    bool seed_explicit;
    size_t residency_samples;
    char output_dir[PATH_MAX];
};

struct residency_report {
    bool query_supported;
    size_t samples;
    size_t on_target;
    size_t other_node;
    size_t errors;
    double target_pct;
};

struct mapping_report {
    uint64_t anon_huge_kb;
    uint64_t kernel_page_kb;
    uint64_t mmu_page_kb;
};

struct thread_quality {
    bool affinity_ok;
    int expected_cpu;
    int start_cpu;
    int end_cpu;
};

struct bw_thread_ctx {
    int thread_id;
    int cpu;
    uint8_t *buffer;
    size_t size;
    traffic_pattern_t pattern;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t checksum;
    uint64_t end_ns;
    struct thread_quality quality;
};

struct lat_thread_ctx {
    int cpu;
    struct node *start;
    struct node *final_ptr;
    uint64_t hops;
    uint64_t end_ns;
    struct thread_quality quality;
};

struct raw_result {
    traffic_pattern_t pattern;
    int threads;
    int repeat;
    uint64_t read_bytes;
    uint64_t write_bytes;
    double elapsed_sec;
    double bandwidth_Bps;
    double latency_ns;
    uint64_t latency_hops;
    long minor_faults;
    long major_faults;
    long voluntary_cs;
    long involuntary_cs;
    bool affinity_verified;
    bool timing_valid;
    bool residency_pre_valid;
    bool residency_post_valid;
    bool valid;
    char invalid_reason[512];
};

struct summary_result {
    int n_total;
    int n_valid;
    double bw_median_Bps;
    double bw_min_Bps;
    double bw_max_Bps;
    double bw_cv_pct;
    double lat_median_ns;
    double lat_min_ns;
    double lat_max_ns;
    double lat_cv_pct;
};

struct benchmark_state {
    struct options opt;
    struct environment env;
    struct residency_report residency_pre;
    struct residency_report residency_post;
    struct mapping_report mapping;
    struct cpu_choice cpus[MAX_CPUS];
    int cpu_count;
    int latency_cpu;
    int bw_cpus[MAX_THREADS];
    int bw_cpu_count;
    void *mem;
    size_t alloc_size;
    size_t lat_size;
    uint8_t *bw_base;
    size_t bw_size;
    struct node *lat_start;
    struct raw_result *raw;
    size_t raw_count;
    size_t raw_capacity;
};

/* ---------- Synchronization shared only within one run ---------- */
static pthread_barrier_t g_start_barrier;
static pthread_barrier_t g_end_barrier;
static _Atomic uint64_t g_start_ns;
static _Atomic uint64_t g_deadline_ns;
static _Atomic int g_abort_run;

/* ---------- Utility ---------- */
static void die(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void die(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ERROR: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(EXIT_FAILURE);
}

static void warnx(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void warnx(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "WARNING: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
}

static uint64_t monotonic_raw_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0)
        die("clock_gettime(CLOCK_MONOTONIC_RAW): %s", strerror(errno));
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void sleep_until_ns(uint64_t deadline)
{
    for (;;) {
        uint64_t now = monotonic_raw_ns();
        if (now >= deadline) return;
        uint64_t rem = deadline - now;
        struct timespec ts = {
            .tv_sec = (time_t)(rem / 1000000000ULL),
            .tv_nsec = (long)(rem % 1000000000ULL)
        };
        if (nanosleep(&ts, &ts) == 0) return;
        if (errno != EINTR) return;
    }
}

static void read_text_file(const char *path, char *out, size_t out_sz)
{
    if (out_sz == 0) return;
    out[0] = '\0';
    FILE *fp = fopen(path, "r");
    if (!fp) return;
    if (fgets(out, (int)out_sz, fp)) out[strcspn(out, "\r\n")] = '\0';
    fclose(fp);
}

static int read_int_file(const char *path, int fallback)
{
    char b[128];
    read_text_file(path, b, sizeof(b));
    if (!b[0]) return fallback;
    char *end = NULL;
    errno = 0;
    long v = strtol(b, &end, 0);
    if (errno || end == b || v < INT_MIN || v > INT_MAX) return fallback;
    return (int)v;
}

static uint64_t parse_size_kib_string(const char *s)
{
    if (!s || !*s) return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || end == s) return 0;
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end == 'K' || *end == 'k') return (uint64_t)v * KIB;
    if (*end == 'M' || *end == 'm') return (uint64_t)v * MIB;
    if (*end == 'G' || *end == 'g') return (uint64_t)v * GIB;
    return (uint64_t)v;
}

static int mkdir_p(const char *path)
{
    char tmp[PATH_MAX];
    if (strlen(path) >= sizeof(tmp)) return -1;
    strcpy(tmp, path);
    size_t len = strlen(tmp);
    if (len == 0) return -1;
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}


static void copy_trunc(char *dst, size_t dst_sz, const char *src)
{
    if (dst_sz == 0) return;
    size_t n = strlen(src);
    if (n >= dst_sz) n = dst_sz - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static int parse_long_strict(const char *s, long minv, long maxv, long *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v < minv || v > maxv) return -1;
    *out = v;
    return 0;
}

static int parse_double_strict(const char *s, double minv, double maxv, double *out)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(s, &end);
    if (errno || end == s || *end != '\0' || !isfinite(v) || v < minv || v > maxv) return -1;
    *out = v;
    return 0;
}

/* ---------- cpulist parser ---------- */
static int parse_id_list(const char *s, int *out, int cap)
{
    int n = 0;
    const char *p = s;
    while (*p) {
        while (isspace((unsigned char)*p) || *p == ',') p++;
        if (!*p) break;
        char *end = NULL;
        errno = 0;
        long a = strtol(p, &end, 10);
        if (errno || end == p || a < 0 || a > INT_MAX) return -1;
        long b = a;
        p = end;
        if (*p == '-') {
            p++;
            errno = 0;
            b = strtol(p, &end, 10);
            if (errno || end == p || b < a || b > INT_MAX) return -1;
            p = end;
        }
        for (long v = a; v <= b; v++) {
            if (n >= cap) return -1;
            out[n++] = (int)v;
        }
        while (isspace((unsigned char)*p)) p++;
        if (*p == ',') p++;
        else if (*p && *p != '\n') return -1;
    }
    return n;
}

static int node_cpu_choices(int node, struct cpu_choice *choices, int cap, char *cpulist_out, size_t cpulist_sz)
{
    char path[PATH_MAX], list[4096];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
    read_text_file(path, list, sizeof(list));
    if (!list[0]) return -1;
    snprintf(cpulist_out, cpulist_sz, "%s", list);

    int ids[MAX_CPUS];
    int n = parse_id_list(list, ids, MAX_CPUS);
    if (n <= 0) return -1;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        return -1;

    struct cpu_choice tmp[MAX_CPUS];
    int tn = 0;
    for (int i = 0; i < n && tn < cap; i++) {
        int c = ids[i];
        if (c >= CPU_SETSIZE || !CPU_ISSET((size_t)c, &allowed)) continue;
        char p1[PATH_MAX], p2[PATH_MAX];
        snprintf(p1, sizeof(p1), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", c);
        snprintf(p2, sizeof(p2), "/sys/devices/system/cpu/cpu%d/topology/core_id", c);
        tmp[tn].cpu = c;
        tmp[tn].package_id = read_int_file(p1, -1);
        tmp[tn].core_id = read_int_file(p2, c);
        tmp[tn].primary_thread = true;
        tn++;
    }
    if (tn == 0) return -1;

    /* Mark first logical CPU of each physical core as primary. */
    for (int i = 0; i < tn; i++) {
        for (int j = 0; j < i; j++) {
            if (tmp[i].package_id == tmp[j].package_id && tmp[i].core_id == tmp[j].core_id) {
                tmp[i].primary_thread = false;
                break;
            }
        }
    }

    int on = 0;
    for (int pass = 0; pass < 2; pass++) {
        bool want_primary = (pass == 0);
        for (int i = 0; i < tn && on < cap; i++) {
            if (tmp[i].primary_thread == want_primary) choices[on++] = tmp[i];
        }
    }
    return on;
}

static bool pin_this_thread(int cpu, struct thread_quality *q)
{
    q->expected_cpu = cpu;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        q->affinity_ok = false;
        q->start_cpu = q->end_cpu = -1;
        return false;
    }
    CPU_SET((size_t)cpu, &set);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    q->affinity_ok = (rc == 0);
    q->start_cpu = sched_getcpu();
    q->end_cpu = q->start_cpu;
    if (rc != 0) return false;
    if (q->start_cpu != cpu) q->affinity_ok = false;
    return q->affinity_ok;
}

/* ---------- NUMA syscalls without libnuma ---------- */
static int highest_possible_node(void)
{
    char b[4096];
    read_text_file("/sys/devices/system/node/possible", b, sizeof(b));
    int ids[MAX_CPUS];
    int n = parse_id_list(b, ids, MAX_CPUS);
    if (n <= 0) return 1023;
    int max = 0;
    for (int i = 0; i < n; i++) if (ids[i] > max) max = ids[i];
    return max;
}

static int bind_memory_to_node(void *addr, size_t len, int node)
{
#if defined(SYS_mbind)
    int maxnode = highest_possible_node() + 1;
    if (node >= maxnode) maxnode = node + 1;
    size_t bits = sizeof(unsigned long) * 8U;
    size_t words = ((size_t)maxnode + bits - 1U) / bits;
    unsigned long *mask = calloc(words, sizeof(*mask));
    if (!mask) return -1;
    mask[(size_t)node / bits] |= 1UL << ((size_t)node % bits);
    long rc = syscall(SYS_mbind, addr, len, MPOL_BIND, mask,
                      (unsigned long)maxnode, MPOL_MF_STRICT);
    int saved = errno;
    free(mask);
    errno = saved;
    return rc == 0 ? 0 : -1;
#else
    (void)addr; (void)len; (void)node;
    errno = ENOSYS;
    return -1;
#endif
}

static struct residency_report verify_residency(void *base, size_t len, int target_node, size_t wanted_samples)
{
    struct residency_report r = {0};
#if defined(SYS_move_pages)
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0 || len < (size_t)ps) return r;
    size_t max_pages = len / (size_t)ps;
    size_t n = wanted_samples < max_pages ? wanted_samples : max_pages;
    if (n == 0) return r;
    void **pages = calloc(n, sizeof(*pages));
    int *status = calloc(n, sizeof(*status));
    if (!pages || !status) { free(pages); free(status); return r; }
    for (size_t i = 0; i < n; i++) {
        size_t page_index = (i * max_pages) / n;
        pages[i] = (uint8_t *)base + page_index * (size_t)ps;
    }
    long rc = syscall(SYS_move_pages, 0, (unsigned long)n, pages, NULL, status, 0);
    if (rc < 0) {
        free(pages); free(status);
        return r;
    }
    r.query_supported = true;
    r.samples = n;
    for (size_t i = 0; i < n; i++) {
        if (status[i] == target_node) r.on_target++;
        else if (status[i] >= 0) r.other_node++;
        else r.errors++;
    }
    r.target_pct = n ? (100.0 * (double)r.on_target / (double)n) : 0.0;
    free(pages); free(status);
#else
    (void)base; (void)len; (void)target_node; (void)wanted_samples;
#endif
    return r;
}

static int read_numa_distance(int cpu_node, int mem_node)
{
    char path[PATH_MAX], b[4096];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/distance", cpu_node);
    read_text_file(path, b, sizeof(b));
    if (!b[0]) return -1;
    int vals[MAX_CPUS];
    int n = 0;
    char *save = NULL;
    for (char *tok = strtok_r(b, " \t", &save); tok && n < MAX_CPUS; tok = strtok_r(NULL, " \t", &save))
        vals[n++] = atoi(tok);
    return (mem_node >= 0 && mem_node < n) ? vals[mem_node] : -1;
}

static uint64_t node_mem_total_bytes(int node)
{
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/meminfo", node);
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    char line[512];
    uint64_t out = 0;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long kb = 0;
        if (strstr(line, "MemTotal") && sscanf(line, "%*[^:]: %llu kB", &kb) == 1) {
            out = (uint64_t)kb * KIB;
            break;
        }
    }
    fclose(fp);
    return out;
}

/* ---------- CXL / environment discovery ---------- */
static bool starts_with(const char *s, const char *pfx)
{
    return strncmp(s, pfx, strlen(pfx)) == 0;
}

static void gather_cpu_info(struct environment *e)
{
    FILE *fp = fopen("/proc/cpuinfo", "r");
    if (!fp) return;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
#if defined(__x86_64__)
        if (!e->cpu_model[0] && starts_with(line, "model name")) {
            char *c = strchr(line, ':'); if (c) snprintf(e->cpu_model, sizeof(e->cpu_model), "%s", c + 2);
        } else if (!e->cpu_microcode[0] && starts_with(line, "microcode")) {
            char *c = strchr(line, ':'); if (c) snprintf(e->cpu_microcode, sizeof(e->cpu_microcode), "%s", c + 2);
        }
#elif defined(__aarch64__)
        if (!e->cpu_model[0] && (starts_with(line, "model name") || starts_with(line, "Processor"))) {
            char *c = strchr(line, ':'); if (c) snprintf(e->cpu_model, sizeof(e->cpu_model), "%s", c + 2);
        }
#endif
        e->cpu_model[strcspn(e->cpu_model, "\r\n")] = '\0';
        e->cpu_microcode[strcspn(e->cpu_microcode, "\r\n")] = '\0';
    }
    fclose(fp);
}

static uint64_t get_llc_bytes(int cpu)
{
    char dirpath[256];
    snprintf(dirpath, sizeof(dirpath), "/sys/devices/system/cpu/cpu%d/cache", cpu);
    DIR *d = opendir(dirpath);
    if (!d) return 0;
    uint64_t best = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (!starts_with(de->d_name, "index")) continue;
        char p[PATH_MAX], type[64], size[64];
        snprintf(p, sizeof(p), "%s/%s/type", dirpath, de->d_name);
        read_text_file(p, type, sizeof(type));
        if (strcmp(type, "Unified") != 0 && strcmp(type, "Data") != 0) continue;
        snprintf(p, sizeof(p), "%s/%s/size", dirpath, de->d_name);
        read_text_file(p, size, sizeof(size));
        uint64_t b = parse_size_kib_string(size);
        if (b > best) best = b;
    }
    closedir(d);
    return best;
}

static void detect_cxl_pmu(struct environment *e)
{
    DIR *d = opendir("/sys/bus/event_sources/devices");
    if (!d) return;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (starts_with(de->d_name, "cxl_pmu_mem")) {
            e->cxl_pmu_present = true;
            break;
        }
    }
    closedir(d);
}


static uint64_t read_u64_text(const char *path, int base, bool *ok)
{
    char b[128];
    read_text_file(path, b, sizeof(b));
    if (!b[0]) { if (ok) *ok = false; return 0; }
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(b, &end, base);
    while (end && *end && isspace((unsigned char)*end)) end++;
    if (errno || !end || end == b || *end) { if (ok) *ok = false; return 0; }
    if (ok) *ok = true;
    return (uint64_t)v;
}

static int region_overlap_with_node(int node, uint64_t region_start, uint64_t region_size)
{
    if (region_size == 0) return 0;
    bool ok = false;
    uint64_t block_size = read_u64_text("/sys/devices/system/memory/block_size_bytes", 16, &ok);
    if (!ok || block_size == 0) return 0;
    char nd[PATH_MAX];
    snprintf(nd, sizeof(nd), "/sys/devices/system/node/node%d", node);
    DIR *d = opendir(nd);
    if (!d) return 0;
    int overlaps = 0;
    struct dirent *de;
    uint64_t region_end = region_start + region_size;
    if (region_end < region_start) region_end = UINT64_MAX;
    while ((de = readdir(d))) {
        if (!starts_with(de->d_name, "memory") || !isdigit((unsigned char)de->d_name[6])) continue;
        char p[PATH_MAX];
        snprintf(p, sizeof(p), "/sys/devices/system/memory/%s/phys_index", de->d_name);
        bool ix_ok = false;
        uint64_t ix = read_u64_text(p, 16, &ix_ok);
        if (!ix_ok) continue;
        uint64_t bstart = ix * block_size;
        uint64_t bend = bstart + block_size;
        if (bend < bstart) bend = UINT64_MAX;
        if (bstart < region_end && bend > region_start) overlaps++;
    }
    closedir(d);
    return overlaps;
}

static void gather_cxl_topology(struct environment *e)
{
    DIR *d = opendir("/sys/bus/cxl/devices");
    if (!d) {
        e->cxl_sysfs_present = false;
        snprintf(e->measurement_scope, sizeof(e->measurement_scope), "numa_node_unverified_as_cxl");
        return;
    }
    e->cxl_sysfs_present = true;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (starts_with(de->d_name, "mem") && isdigit((unsigned char)de->d_name[3])) {
            if (e->cxl_memdev_count >= MAX_CXL_MEMDEVS) continue;
            char p[PATH_MAX];
            snprintf(p, sizeof(p), "/sys/bus/cxl/devices/%s/numa_node", de->d_name);
            int nn = read_int_file(p, -1);
            if (nn != e->cxl_node) continue;
            struct cxl_memdev_info *m = &e->memdevs[e->cxl_memdev_count++];
            copy_trunc(m->name, sizeof(m->name), de->d_name);
            m->numa_node = nn;
            snprintf(m->sysfs_path, sizeof(m->sysfs_path), "/sys/bus/cxl/devices/%s", de->d_name);
            char resolved[PATH_MAX];
            if (realpath(m->sysfs_path, resolved)) snprintf(m->canonical_path, sizeof(m->canonical_path), "%s", resolved);
            snprintf(p, sizeof(p), "%s/serial", m->sysfs_path); read_text_file(p, m->serial, sizeof(m->serial));
            snprintf(p, sizeof(p), "%s/firmware_version", m->sysfs_path); read_text_file(p, m->firmware_version, sizeof(m->firmware_version));
            snprintf(p, sizeof(p), "%s/ram", m->sysfs_path); read_text_file(p, m->ram, sizeof(m->ram));
            snprintf(p, sizeof(p), "%s/pmem", m->sysfs_path); read_text_file(p, m->pmem, sizeof(m->pmem));
        }
    }
    rewinddir(d);
    while ((de = readdir(d))) {
        if (starts_with(de->d_name, "region") && isdigit((unsigned char)de->d_name[6])) {
            if (e->cxl_region_count >= MAX_CXL_REGIONS) continue;
            struct cxl_region_info *r = &e->regions[e->cxl_region_count++];
            copy_trunc(r->name, sizeof(r->name), de->d_name);
            char base[512], p[1024];
            snprintf(base, sizeof(base), "/sys/bus/cxl/devices/%s", de->d_name);
            snprintf(p, sizeof(p), "%s/mode", base); read_text_file(p, r->mode, sizeof(r->mode));
            snprintf(p, sizeof(p), "%s/resource", base); read_text_file(p, r->resource, sizeof(r->resource));
            snprintf(p, sizeof(p), "%s/size", base); read_text_file(p, r->size, sizeof(r->size));
            snprintf(p, sizeof(p), "%s/interleave_ways", base); read_text_file(p, r->interleave_ways, sizeof(r->interleave_ways));
            snprintf(p, sizeof(p), "%s/interleave_granularity", base); read_text_file(p, r->interleave_granularity, sizeof(r->interleave_granularity));
            DIR *rd = opendir(base);
            if (rd) {
                struct dirent *x;
                while ((x = readdir(rd))) {
                    if (!starts_with(x->d_name, "target") || !isdigit((unsigned char)x->d_name[6])) continue;
                    if (r->target_count >= MAX_REGION_TARGETS) continue;
                    char linkp[1024], target[PATH_MAX];
                    snprintf(linkp, sizeof(linkp), "%s/%s", base, x->d_name);
                    ssize_t n = readlink(linkp, target, sizeof(target) - 1);
                    if (n > 0) {
                        target[n] = '\0';
                        char *slash = strrchr(target, '/');
                        const char *tn = slash ? slash + 1 : target;
                        copy_trunc(r->targets[r->target_count], sizeof(r->targets[0]), tn);
                        r->target_count++;
                    } else {
                        read_text_file(linkp, r->targets[r->target_count], sizeof(r->targets[0]));
                        if (r->targets[r->target_count][0]) r->target_count++;
                    }
                }
                closedir(rd);
            }
            bool rs_ok = false, rz_ok = false;
            uint64_t rs = 0, rz = 0;
            char rp[PATH_MAX];
            snprintf(rp, sizeof(rp), "%s/resource", base); rs = read_u64_text(rp, 0, &rs_ok);
            snprintf(rp, sizeof(rp), "%s/size", base); rz = read_u64_text(rp, 0, &rz_ok);
            if (rs_ok && rz_ok) {
                r->overlapping_memory_blocks = region_overlap_with_node(e->cxl_node, rs, rz);
                r->overlaps_target_node = r->overlapping_memory_blocks > 0;
                if (r->overlaps_target_node) e->cxl_node_overlaps_region = true;
            }
        }
    }
    closedir(d);
    e->cxl_node_has_memdev = e->cxl_memdev_count > 0;

    /* Conservative identity rule: one memdev on node plus a one-way RAM region is
     * sufficient to label the run single-device-candidate, but sysfs alone does not
     * prove endpoint<->memdev mapping on every kernel. Keep scope conservative. */
    bool one_way_region = false;
    for (int i = 0; i < e->cxl_region_count; i++) {
        if (strcmp(e->regions[i].interleave_ways, "1") == 0 && strcmp(e->regions[i].mode, "ram") == 0) {
            one_way_region = true;
            break;
        }
    }
    e->cxl_identity_proven_single_device = (e->cxl_memdev_count == 1 && one_way_region && e->cxl_node_overlaps_region);
    if (!e->cxl_node_overlaps_region && !e->cxl_node_has_memdev)
        snprintf(e->measurement_scope, sizeof(e->measurement_scope), "numa_node_not_correlated_to_cxl");
    else if (e->cxl_identity_proven_single_device)
        snprintf(e->measurement_scope, sizeof(e->measurement_scope), "cpu_to_cxl_node_single_memdev_candidate");
    else
        snprintf(e->measurement_scope, sizeof(e->measurement_scope), "cpu_to_cxl_node_or_region_aggregate");
}


static void gather_binary_sha256(char *out, size_t out_sz)
{
    out[0] = '\0';
    FILE *p = popen("sha256sum /proc/self/exe 2>/dev/null", "r");
    if (!p) return;
    char b[256];
    if (fgets(b, sizeof(b), p)) {
        char *sp = strchr(b, ' ');
        if (sp) *sp = '\0';
        copy_trunc(out, out_sz, b);
    }
    (void)pclose(p);
}

static void gather_distro(char *out, size_t out_sz)
{
    out[0] = '\0';
    FILE *fp = fopen("/etc/os-release", "r");
    if (!fp) return;
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (!starts_with(line, "PRETTY_NAME=")) continue;
        char *v = strchr(line, '=');
        if (!v) break;
        v++;
        v[strcspn(v, "\r\n")] = '\0';
        size_t n = strlen(v);
        if (n >= 2 && v[0] == '"' && v[n-1] == '"') { v[n-1] = '\0'; v++; }
        copy_trunc(out, out_sz, v);
        break;
    }
    fclose(fp);
}

static void gather_environment(struct benchmark_state *s)
{
    struct environment *e = &s->env;
    memset(e, 0, sizeof(*e));
    e->cpu_node = s->opt.cpu_node;
    e->cxl_node = s->opt.cxl_node;
    time_t now = time(NULL);
    struct tm tm;
    gmtime_r(&now, &tm);
    strftime(e->timestamp, sizeof(e->timestamp), "%Y-%m-%dT%H:%M:%SZ", &tm);
    struct utsname u;
    if (uname(&u) == 0) {
        snprintf(e->hostname, sizeof(e->hostname), "%s", u.nodename);
        snprintf(e->kernel_release, sizeof(e->kernel_release), "%s", u.release);
        snprintf(e->architecture, sizeof(e->architecture), "%s", u.machine);
    }
    gather_cpu_info(e);
    read_text_file("/sys/class/dmi/id/bios_version", e->bios_version, sizeof(e->bios_version));
    read_text_file("/sys/class/dmi/id/bios_date", e->bios_date, sizeof(e->bios_date));
    read_text_file("/proc/cmdline", e->kernel_cmdline, sizeof(e->kernel_cmdline));
    e->numa_balancing = read_int_file("/proc/sys/kernel/numa_balancing", -1);
    read_text_file("/sys/kernel/mm/transparent_hugepage/enabled", e->thp_status, sizeof(e->thp_status));
    read_text_file("/proc/self/status", e->mems_allowed_list, 1); /* overwritten below if found */
    FILE *sf = fopen("/proc/self/status", "r");
    if (sf) {
        char line[2048];
        while (fgets(line, sizeof(line), sf)) {
            if (starts_with(line, "Mems_allowed_list:")) {
                char *c = strchr(line, ':');
                if (c) {
                    c++; while (*c && isspace((unsigned char)*c)) c++;
                    snprintf(e->mems_allowed_list, sizeof(e->mems_allowed_list), "%s", c);
                    e->mems_allowed_list[strcspn(e->mems_allowed_list, "\r\n")] = '\0';
                }
            }
        }
        fclose(sf);
    }
#ifdef __VERSION__
    snprintf(e->compiler, sizeof(e->compiler), "%s", __VERSION__);
#else
    snprintf(e->compiler, sizeof(e->compiler), "unknown");
#endif
    copy_trunc(e->build_flags, sizeof(e->build_flags), CXL_BENCH_BUILD_FLAGS);
    copy_trunc(e->git_commit, sizeof(e->git_commit), CXL_BENCH_GIT_COMMIT);
    gather_binary_sha256(e->binary_sha256, sizeof(e->binary_sha256));
    gather_distro(e->distro, sizeof(e->distro));
#if defined(__x86_64__)
    snprintf(e->isa_backend, sizeof(e->isa_backend), "x86_64-sse2-full-line; NTW=_mm_stream_si128 x4");
#elif defined(__aarch64__)
    snprintf(e->isa_backend, sizeof(e->isa_backend), "aarch64-advSIMD-full-line; NTW=STNP-hint x4");
#endif
    e->numa_distance = read_numa_distance(e->cpu_node, e->cxl_node);
    e->cxl_node_mem_total_bytes = node_mem_total_bytes(e->cxl_node);
    gather_cxl_topology(e);
    detect_cxl_pmu(e);
}

/* ---------- Mapping inspection ---------- */
static struct mapping_report inspect_mapping(void *addr)
{
    struct mapping_report r = {0};
    FILE *fp = fopen("/proc/self/smaps", "r");
    if (!fp) return r;
    char line[512];
    uintptr_t needle = (uintptr_t)addr;
    bool in = false;
    while (fgets(line, sizeof(line), fp)) {
        unsigned long long a = 0, b = 0;
        if (sscanf(line, "%llx-%llx", &a, &b) == 2) {
            in = needle >= (uintptr_t)a && needle < (uintptr_t)b;
            continue;
        }
        if (!in) continue;
        unsigned long long kb = 0;
        if (sscanf(line, "AnonHugePages: %llu kB", &kb) == 1) r.anon_huge_kb = (uint64_t)kb;
        else if (sscanf(line, "KernelPageSize: %llu kB", &kb) == 1) r.kernel_page_kb = (uint64_t)kb;
        else if (sscanf(line, "MMUPageSize: %llu kB", &kb) == 1) r.mmu_page_kb = (uint64_t)kb;
    }
    fclose(fp);
    return r;
}

/* ---------- Deterministic PRNG / pointer chase ---------- */
static uint64_t splitmix64_next(uint64_t *x)
{
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static struct node *build_single_chase(void *mem, size_t size, uint64_t seed)
{
    size_t n = size / sizeof(struct node);
    if (n < 2) die("latency working set is too small");
    struct node *nodes = mem;
    size_t *idx = malloc(n * sizeof(*idx));
    if (!idx) die("cannot allocate pointer-shuffle index array (%zu entries)", n);
    for (size_t i = 0; i < n; i++) idx[i] = i;
    uint64_t st = seed;
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(splitmix64_next(&st) % (i + 1));
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    for (size_t i = 0; i < n - 1; i++) nodes[idx[i]].next = &nodes[idx[i + 1]];
    nodes[idx[n - 1]].next = &nodes[idx[0]];
    struct node *start = &nodes[idx[0]];
    free(idx);
    return start;
}

/* ---------- ISA kernels ---------- */
#if defined(__x86_64__)
static inline uint64_t read_line_accum(const uint8_t *p, __m128i *acc)
{
    __m128i a = _mm_load_si128((const __m128i *)(p + 0));
    __m128i b = _mm_load_si128((const __m128i *)(p + 16));
    __m128i c = _mm_load_si128((const __m128i *)(p + 32));
    __m128i d = _mm_load_si128((const __m128i *)(p + 48));
    *acc = _mm_xor_si128(*acc, _mm_xor_si128(_mm_xor_si128(a, b), _mm_xor_si128(c, d)));
    return 64;
}
static inline void nt_write_line(uint8_t *p)
{
    const __m128i z = _mm_setzero_si128();
    _mm_stream_si128((__m128i *)(p + 0), z);
    _mm_stream_si128((__m128i *)(p + 16), z);
    _mm_stream_si128((__m128i *)(p + 32), z);
    _mm_stream_si128((__m128i *)(p + 48), z);
}
static inline void normal_write_line(uint8_t *p)
{
    const __m128i z = _mm_setzero_si128();
    _mm_store_si128((__m128i *)(p + 0), z);
    _mm_store_si128((__m128i *)(p + 16), z);
    _mm_store_si128((__m128i *)(p + 32), z);
    _mm_store_si128((__m128i *)(p + 48), z);
}
static inline uint64_t reduce_acc(__m128i a)
{
    uint64_t v[2];
    _mm_storeu_si128((__m128i *)v, a);
    return v[0] ^ v[1];
}
static inline void store_fence(void) { _mm_sfence(); }
#elif defined(__aarch64__)
static inline uint64_t read_line_accum(const uint8_t *p, uint64x2_t *acc)
{
    uint64x2_t a = vld1q_u64((const uint64_t *)(p + 0));
    uint64x2_t b = vld1q_u64((const uint64_t *)(p + 16));
    uint64x2_t c = vld1q_u64((const uint64_t *)(p + 32));
    uint64x2_t d = vld1q_u64((const uint64_t *)(p + 48));
    *acc = veorq_u64(*acc, veorq_u64(veorq_u64(a,b), veorq_u64(c,d)));
    return 64;
}
static inline void nt_write_line(uint8_t *p)
{
    __asm__ volatile(
        "stnp xzr, xzr, [%0, #0]\n\t"
        "stnp xzr, xzr, [%0, #16]\n\t"
        "stnp xzr, xzr, [%0, #32]\n\t"
        "stnp xzr, xzr, [%0, #48]\n\t"
        : : "r"(p) : "memory");
}
static inline void normal_write_line(uint8_t *p)
{
    __asm__ volatile(
        "stp xzr, xzr, [%0, #0]\n\t"
        "stp xzr, xzr, [%0, #16]\n\t"
        "stp xzr, xzr, [%0, #32]\n\t"
        "stp xzr, xzr, [%0, #48]\n\t"
        : : "r"(p) : "memory");
}
static inline uint64_t reduce_acc(uint64x2_t a)
{
    return vgetq_lane_u64(a, 0) ^ vgetq_lane_u64(a, 1);
}
static inline void store_fence(void) { __asm__ volatile("dsb sy" ::: "memory"); }
#endif

static inline void consume_u64(uint64_t v)
{
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : : "r"(v) : "memory");
#else
    (void)v;
#endif
}

static inline void consume_ptr(const void *p)
{
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile("" : : "r"(p) : "memory");
#else
    (void)p;
#endif
}

/* ---------- Workers ---------- */
static void wait_for_start(uint64_t *start, uint64_t *deadline)
{
    while (atomic_load_explicit(&g_start_ns, memory_order_acquire) == 0) {
        if (atomic_load_explicit(&g_abort_run, memory_order_relaxed)) break;
        __asm__ volatile("" ::: "memory");
    }
    *start = atomic_load_explicit(&g_start_ns, memory_order_acquire);
    *deadline = atomic_load_explicit(&g_deadline_ns, memory_order_acquire);
}

static void *bandwidth_worker(void *arg)
{
    struct bw_thread_ctx *c = arg;
    pin_this_thread(c->cpu, &c->quality);
    pthread_barrier_wait(&g_start_barrier);
    uint64_t start, deadline;
    wait_for_start(&start, &deadline);
    (void)start;
    if (atomic_load_explicit(&g_abort_run, memory_order_relaxed)) {
        c->end_ns = monotonic_raw_ns();
        pthread_barrier_wait(&g_end_barrier);
        return NULL;
    }

    size_t lines = c->size / CACHE_LINE_SIZE;
    uint64_t rb = 0, wb = 0;
#if defined(__x86_64__)
    __m128i acc = _mm_setzero_si128();
#elif defined(__aarch64__)
    uint64x2_t acc = vdupq_n_u64(0);
#endif

    while (true) {
        for (size_t base = 0; base < lines; base += BW_TIME_CHECK_LINES) {
            size_t end = base + BW_TIME_CHECK_LINES;
            if (end > lines) end = lines;
            for (size_t i = base; i < end; i++) {
                uint8_t *p = c->buffer + i * CACHE_LINE_SIZE;
                switch (c->pattern) {
                    case TP_R:
                        rb += read_line_accum(p, &acc);
                        break;
                    case TP_NTW:
                        nt_write_line(p); wb += 64; break;
                    case TP_R2_NTW1:
                        if ((i % 3U) < 2U) rb += read_line_accum(p, &acc);
                        else { nt_write_line(p); wb += 64; }
                        break;
                    case TP_R1_NTW1:
                        if ((i & 1U) == 0U) rb += read_line_accum(p, &acc);
                        else { nt_write_line(p); wb += 64; }
                        break;
                    case TP_R2_W1:
                        if ((i % 3U) < 2U) rb += read_line_accum(p, &acc);
                        else { normal_write_line(p); wb += 64; }
                        break;
                    default: break;
                }
            }
            if (monotonic_raw_ns() >= deadline) goto done;
        }
    }
done:
    store_fence();
    c->checksum = reduce_acc(acc);
    consume_u64(c->checksum);
    c->read_bytes = rb;
    c->write_bytes = wb;
    c->quality.end_cpu = sched_getcpu();
    if (c->quality.end_cpu != c->quality.expected_cpu) c->quality.affinity_ok = false;
    c->end_ns = monotonic_raw_ns();
    pthread_barrier_wait(&g_end_barrier);
    return NULL;
}

static void *latency_worker(void *arg)
{
    struct lat_thread_ctx *c = arg;
    pin_this_thread(c->cpu, &c->quality);
    struct node *p = c->start;
    pthread_barrier_wait(&g_start_barrier);
    uint64_t start, deadline;
    wait_for_start(&start, &deadline);
    (void)start;
    if (atomic_load_explicit(&g_abort_run, memory_order_relaxed)) {
        c->final_ptr = p;
        c->end_ns = monotonic_raw_ns();
        pthread_barrier_wait(&g_end_barrier);
        return NULL;
    }
    uint64_t hops = 0;
    while (true) {
        for (uint64_t i = 0; i < LAT_TIME_CHECK_HOPS; i++) {
            p = p->next; /* exactly one dependent chain */
        }
        hops += LAT_TIME_CHECK_HOPS;
        if (monotonic_raw_ns() >= deadline) break;
    }
    consume_ptr(p);
    c->final_ptr = p;
    c->hops = hops;
    c->quality.end_cpu = sched_getcpu();
    if (c->quality.end_cpu != c->quality.expected_cpu) c->quality.affinity_ok = false;
    c->end_ns = monotonic_raw_ns();
    pthread_barrier_wait(&g_end_barrier);
    return NULL;
}

/* ---------- Raw result management ---------- */
static void push_raw(struct benchmark_state *s, const struct raw_result *r)
{
    if (s->raw_count == s->raw_capacity) {
        size_t nc = s->raw_capacity ? s->raw_capacity * 2U : 256U;
        struct raw_result *nr = realloc(s->raw, nc * sizeof(*nr));
        if (!nr) die("out of memory growing result array");
        s->raw = nr;
        s->raw_capacity = nc;
    }
    s->raw[s->raw_count++] = *r;
}

static void append_reason(char *dst, size_t sz, const char *reason)
{
    if (!reason || !*reason) return;
    size_t n = strlen(dst);
    if (n && n + 2 < sz) { dst[n++] = ';'; dst[n++] = ' '; dst[n] = '\0'; }
    if (n < sz - 1) strncat(dst, reason, sz - n - 1);
}

static void rusage_delta(const struct rusage *a, const struct rusage *b,
                         long *minflt, long *majflt, long *nvcsw, long *nivcsw)
{
    *minflt = b->ru_minflt - a->ru_minflt;
    *majflt = b->ru_majflt - a->ru_majflt;
    *nvcsw = b->ru_nvcsw - a->ru_nvcsw;
    *nivcsw = b->ru_nivcsw - a->ru_nivcsw;
}

static struct raw_result run_once(struct benchmark_state *s, traffic_pattern_t pattern,
                                  int threads, int repeat, double seconds, bool record)
{
    struct raw_result rr;
    memset(&rr, 0, sizeof(rr));
    rr.pattern = pattern;
    rr.threads = threads;
    rr.repeat = repeat;

    if (threads > s->bw_cpu_count) die("requested %d bandwidth threads, only %d usable CPUs", threads, s->bw_cpu_count);

    size_t chunk = (s->bw_size / (size_t)threads) & ~(size_t)(CACHE_LINE_SIZE - 1U);
    if (chunk < CACHE_LINE_SIZE * 1024U) die("per-thread working set too small");

    pthread_t *tids = calloc((size_t)threads, sizeof(*tids));
    struct bw_thread_ctx *bc = calloc((size_t)threads, sizeof(*bc));
    if (!tids || !bc) die("cannot allocate worker metadata");
    pthread_t lat_tid;
    struct lat_thread_ctx lc;
    memset(&lc, 0, sizeof(lc));
    lc.cpu = s->latency_cpu;
    lc.start = s->lat_start;

    int participants = threads + 2; /* BW + latency + main */
    if (pthread_barrier_init(&g_start_barrier, NULL, (unsigned)participants) != 0 ||
        pthread_barrier_init(&g_end_barrier, NULL, (unsigned)participants) != 0)
        die("pthread_barrier_init failed");
    atomic_store(&g_start_ns, 0);
    atomic_store(&g_deadline_ns, 0);
    atomic_store(&g_abort_run, 0);

    if (pthread_create(&lat_tid, NULL, latency_worker, &lc) != 0) die("pthread_create latency failed");
    for (int i = 0; i < threads; i++) {
        bc[i].thread_id = i;
        bc[i].cpu = s->bw_cpus[i];
        bc[i].buffer = s->bw_base + (size_t)i * chunk;
        bc[i].size = chunk;
        bc[i].pattern = pattern;
        if (pthread_create(&tids[i], NULL, bandwidth_worker, &bc[i]) != 0)
            die("pthread_create bandwidth[%d] failed", i);
    }

    pthread_barrier_wait(&g_start_barrier);
    bool aff_ok = lc.quality.affinity_ok;
    for (int i = 0; i < threads; i++) aff_ok = aff_ok && bc[i].quality.affinity_ok;
    if (!aff_ok) atomic_store(&g_abort_run, 1);

    struct rusage ru0, ru1;
    getrusage(RUSAGE_SELF, &ru0);
    uint64_t start = monotonic_raw_ns();
    uint64_t duration_ns = (uint64_t)(seconds * 1000000000.0);
    uint64_t deadline = start + duration_ns;
    atomic_store_explicit(&g_deadline_ns, deadline, memory_order_release);
    atomic_store_explicit(&g_start_ns, start, memory_order_release);

    sleep_until_ns(deadline);
    pthread_barrier_wait(&g_end_barrier);
    uint64_t end = monotonic_raw_ns();
    getrusage(RUSAGE_SELF, &ru1);

    pthread_join(lat_tid, NULL);
    for (int i = 0; i < threads; i++) pthread_join(tids[i], NULL);
    pthread_barrier_destroy(&g_start_barrier);
    pthread_barrier_destroy(&g_end_barrier);

    rr.affinity_verified = lc.quality.affinity_ok;
    uint64_t bytes = 0, read_bytes = 0, write_bytes = 0;
    uint64_t max_worker_end = lc.end_ns;
    for (int i = 0; i < threads; i++) {
        rr.affinity_verified = rr.affinity_verified && bc[i].quality.affinity_ok;
        read_bytes += bc[i].read_bytes;
        write_bytes += bc[i].write_bytes;
        if (bc[i].end_ns > max_worker_end) max_worker_end = bc[i].end_ns;
    }
    bytes = read_bytes + write_bytes;
    rr.read_bytes = read_bytes;
    rr.write_bytes = write_bytes;
    uint64_t effective_end = max_worker_end > start ? max_worker_end : end;
    rr.elapsed_sec = (double)(effective_end - start) / 1e9;
    rr.bandwidth_Bps = rr.elapsed_sec > 0.0 ? (double)bytes / rr.elapsed_sec : 0.0;
    double lat_elapsed = lc.end_ns > start ? (double)(lc.end_ns - start) : 0.0;
    rr.latency_hops = lc.hops;
    rr.latency_ns = lc.hops ? lat_elapsed / (double)lc.hops : 0.0;
    rusage_delta(&ru0, &ru1, &rr.minor_faults, &rr.major_faults, &rr.voluntary_cs, &rr.involuntary_cs);

    double target = seconds;
    double err_pct = target > 0.0 ? fabs(rr.elapsed_sec - target) * 100.0 / target : 100.0;
    rr.timing_valid = err_pct <= 2.0; /* block-granularity overrun tolerance */
    rr.residency_pre_valid = s->residency_pre.query_supported &&
                             s->residency_pre.other_node == 0 && s->residency_pre.errors == 0 &&
                             s->residency_pre.on_target == s->residency_pre.samples;
    rr.residency_post_valid = true; /* finalized after suite; JSON carries final gate too */

    rr.valid = true;
    if (!rr.affinity_verified) { rr.valid = false; append_reason(rr.invalid_reason, sizeof(rr.invalid_reason), "CPU affinity verification failed"); }
    if (!rr.timing_valid) { rr.valid = false; append_reason(rr.invalid_reason, sizeof(rr.invalid_reason), "measurement duration error >2%"); }
    if (!rr.residency_pre_valid) { rr.valid = false; append_reason(rr.invalid_reason, sizeof(rr.invalid_reason), "pre-run NUMA residency is not 100% verified on target node"); }
    if (rr.major_faults != 0) { rr.valid = false; append_reason(rr.invalid_reason, sizeof(rr.invalid_reason), "major page fault during measurement"); }
    if (atomic_load(&g_abort_run)) { rr.valid = false; append_reason(rr.invalid_reason, sizeof(rr.invalid_reason), "run aborted before measurement"); }

    if (record) push_raw(s, &rr);
    free(tids); free(bc);
    return rr;
}

/* ---------- Statistics ---------- */
static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void calc_stats(double *v, int n, double *median, double *minv, double *maxv, double *cv)
{
    if (n <= 0) { *median = *minv = *maxv = *cv = NAN; return; }
    qsort(v, (size_t)n, sizeof(*v), cmp_double);
    *minv = v[0]; *maxv = v[n - 1];
    *median = (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += v[i];
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double d = v[i] - mean; var += d*d; }
    var /= (double)n;
    *cv = mean != 0.0 ? (sqrt(var) / mean) * 100.0 : 0.0;
}

static struct summary_result summarize(const struct benchmark_state *s, traffic_pattern_t p, int t)
{
    struct summary_result z = {0};
    double *bw = calloc((size_t)s->opt.repeats, sizeof(*bw));
    double *lat = calloc((size_t)s->opt.repeats, sizeof(*lat));
    if (!bw || !lat) die("out of memory in summarize");
    int n = 0;
    for (size_t i = 0; i < s->raw_count; i++) {
        const struct raw_result *r = &s->raw[i];
        if (r->pattern != p || r->threads != t) continue;
        z.n_total++;
        if (r->valid && s->residency_post.query_supported &&
            s->residency_post.other_node == 0 && s->residency_post.errors == 0 &&
            s->residency_post.on_target == s->residency_post.samples) {
            bw[n] = r->bandwidth_Bps;
            lat[n] = r->latency_ns;
            n++; z.n_valid++;
        }
    }
    calc_stats(bw, n, &z.bw_median_Bps, &z.bw_min_Bps, &z.bw_max_Bps, &z.bw_cv_pct);
    calc_stats(lat, n, &z.lat_median_ns, &z.lat_min_ns, &z.lat_max_ns, &z.lat_cv_pct);
    free(bw); free(lat);
    return z;
}

/* ---------- JSON ---------- */
static void json_string(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)(s ? s : ""); *p; p++) {
        switch (*p) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", *p);
                else fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

static const char *page_mode_name(page_mode_t m)
{
    return m == PAGE_HUGETLB ? "hugetlb" : m == PAGE_THP ? "thp" : "base";
}

static void write_report(struct benchmark_state *s, const char *run_dir)
{
    char final[PATH_MAX], tmp[PATH_MAX];
    snprintf(final, sizeof(final), "%s/report.json", run_dir);
    snprintf(tmp, sizeof(tmp), "%s/report.json.tmp", run_dir);
    FILE *fp = fopen(tmp, "w");
    if (!fp) die("open %s: %s", tmp, strerror(errno));
    struct environment *e = &s->env;

    fprintf(fp, "{\n  \"schema_version\": "); json_string(fp, SCHEMA_VERSION);
    fprintf(fp, ",\n  \"metadata\": {\n    \"tool_version\": "); json_string(fp, VERSION);
    fprintf(fp, ",\n    \"timestamp_utc\": "); json_string(fp, e->timestamp);
    fprintf(fp, ",\n    \"v1_0_baseline_compatible\": false,\n    \"measurement_scope\": "); json_string(fp, e->measurement_scope);
    fprintf(fp, "\n  },\n");

    fprintf(fp, "  \"configuration\": {\n");
    fprintf(fp, "    \"cpu_node\": %d, \"cxl_node\": %d, \"max_threads\": %d,\n", s->opt.cpu_node, s->opt.cxl_node, s->opt.max_threads);
    fprintf(fp, "    \"total_bytes\": %zu, \"latency_bytes\": %zu, \"bandwidth_bytes\": %zu,\n", s->alloc_size, s->lat_size, s->bw_size);
    fprintf(fp, "    \"page_mode\": "); json_string(fp, page_mode_name(s->opt.page_mode));
    fprintf(fp, ", \"duration_sec\": %.6f, \"warmup_sec\": %.6f, \"repeats\": %d,\n", s->opt.duration_sec, s->opt.warmup_sec, s->opt.repeats);
    fprintf(fp, "    \"seed\": %llu, \"residency_samples_requested\": %zu\n  },\n", (unsigned long long)s->opt.seed, s->opt.residency_samples);

    fprintf(fp, "  \"environment\": {\n");
    fprintf(fp, "    \"hostname\": "); json_string(fp, e->hostname); fprintf(fp, ",\n");
    fprintf(fp, "    \"kernel_release\": "); json_string(fp, e->kernel_release); fprintf(fp, ",\n");
    fprintf(fp, "    \"distro\": "); json_string(fp, e->distro); fprintf(fp, ",\n");
    fprintf(fp, "    \"architecture\": "); json_string(fp, e->architecture); fprintf(fp, ",\n");
    fprintf(fp, "    \"cpu_model\": "); json_string(fp, e->cpu_model); fprintf(fp, ",\n");
    fprintf(fp, "    \"cpu_microcode\": "); json_string(fp, e->cpu_microcode); fprintf(fp, ",\n");
    fprintf(fp, "    \"bios_version\": "); json_string(fp, e->bios_version); fprintf(fp, ",\n");
    fprintf(fp, "    \"bios_date\": "); json_string(fp, e->bios_date); fprintf(fp, ",\n");
    fprintf(fp, "    \"kernel_cmdline\": "); json_string(fp, e->kernel_cmdline); fprintf(fp, ",\n");
    fprintf(fp, "    \"compiler\": "); json_string(fp, e->compiler); fprintf(fp, ",\n");
    fprintf(fp, "    \"build_flags\": "); json_string(fp, e->build_flags); fprintf(fp, ",\n");
    fprintf(fp, "    \"git_commit\": "); json_string(fp, e->git_commit); fprintf(fp, ",\n");
    fprintf(fp, "    \"binary_sha256\": "); json_string(fp, e->binary_sha256); fprintf(fp, ",\n");
    fprintf(fp, "    \"isa_backend\": "); json_string(fp, e->isa_backend); fprintf(fp, ",\n");
    fprintf(fp, "    \"numa_balancing\": %d,\n", e->numa_balancing);
    fprintf(fp, "    \"cpu_governor\": "); json_string(fp, e->cpu_governor); fprintf(fp, ",\n");
    fprintf(fp, "    \"thp_status\": "); json_string(fp, e->thp_status); fprintf(fp, ",\n");
    fprintf(fp, "    \"cpu_node_cpulist\": "); json_string(fp, e->cpu_cpulist); fprintf(fp, ",\n");
    fprintf(fp, "    \"mems_allowed_list\": "); json_string(fp, e->mems_allowed_list); fprintf(fp, ",\n");
    fprintf(fp, "    \"numa_distance\": %d, \"cxl_node_mem_total_bytes\": %llu, \"llc_bytes\": %llu\n  },\n",
            e->numa_distance, (unsigned long long)e->cxl_node_mem_total_bytes, (unsigned long long)e->llc_bytes);

    fprintf(fp, "  \"cpu_placement\": {\n    \"latency_cpu\": %d,\n    \"bandwidth_cpus\": [", s->latency_cpu);
    for (int i = 0; i < s->bw_cpu_count; i++) fprintf(fp, "%s%d", i ? ", " : "", s->bw_cpus[i]);
    fprintf(fp, "]\n  },\n");

    fprintf(fp, "  \"memory_mapping\": {\n    \"anon_huge_kb\": %llu, \"kernel_page_kb\": %llu, \"mmu_page_kb\": %llu,\n",
            (unsigned long long)s->mapping.anon_huge_kb, (unsigned long long)s->mapping.kernel_page_kb, (unsigned long long)s->mapping.mmu_page_kb);
    fprintf(fp, "    \"residency_pre\": {\"supported\": %s, \"samples\": %zu, \"on_target\": %zu, \"other_node\": %zu, \"errors\": %zu, \"target_pct\": %.6f},\n",
            s->residency_pre.query_supported ? "true" : "false", s->residency_pre.samples, s->residency_pre.on_target, s->residency_pre.other_node, s->residency_pre.errors, s->residency_pre.target_pct);
    fprintf(fp, "    \"residency_post\": {\"supported\": %s, \"samples\": %zu, \"on_target\": %zu, \"other_node\": %zu, \"errors\": %zu, \"target_pct\": %.6f}\n  },\n",
            s->residency_post.query_supported ? "true" : "false", s->residency_post.samples, s->residency_post.on_target, s->residency_post.other_node, s->residency_post.errors, s->residency_post.target_pct);

    fprintf(fp, "  \"cxl_topology\": {\n    \"sysfs_present\": %s, \"node_has_memdev\": %s, \"node_overlaps_region_hpa\": %s, \"single_device_candidate\": %s, \"cxl_pmu_present\": %s,\n",
            e->cxl_sysfs_present ? "true" : "false", e->cxl_node_has_memdev ? "true" : "false", e->cxl_node_overlaps_region ? "true" : "false", e->cxl_identity_proven_single_device ? "true" : "false", e->cxl_pmu_present ? "true" : "false");
    fprintf(fp, "    \"memdevs_on_target_node\": [\n");
    for (int i = 0; i < e->cxl_memdev_count; i++) {
        struct cxl_memdev_info *m = &e->memdevs[i];
        fprintf(fp, "      {\"name\": "); json_string(fp, m->name);
        fprintf(fp, ", \"numa_node\": %d, \"serial\": ", m->numa_node); json_string(fp, m->serial);
        fprintf(fp, ", \"firmware_version\": "); json_string(fp, m->firmware_version);
        fprintf(fp, ", \"ram\": "); json_string(fp, m->ram);
        fprintf(fp, ", \"pmem\": "); json_string(fp, m->pmem);
        fprintf(fp, ", \"canonical_path\": "); json_string(fp, m->canonical_path);
        fprintf(fp, "}%s\n", i + 1 == e->cxl_memdev_count ? "" : ",");
    }
    fprintf(fp, "    ],\n    \"regions_visible\": [\n");
    for (int i = 0; i < e->cxl_region_count; i++) {
        struct cxl_region_info *r = &e->regions[i];
        fprintf(fp, "      {\"name\": "); json_string(fp, r->name);
        fprintf(fp, ", \"mode\": "); json_string(fp, r->mode);
        fprintf(fp, ", \"resource\": "); json_string(fp, r->resource);
        fprintf(fp, ", \"size\": "); json_string(fp, r->size);
        fprintf(fp, ", \"interleave_ways\": "); json_string(fp, r->interleave_ways);
        fprintf(fp, ", \"interleave_granularity\": "); json_string(fp, r->interleave_granularity);
        fprintf(fp, ", \"overlaps_target_node\": %s, \"overlapping_memory_blocks\": %d", r->overlaps_target_node ? "true" : "false", r->overlapping_memory_blocks);
        fprintf(fp, ", \"targets\": [");
        for (int j = 0; j < r->target_count; j++) { if (j) fputs(", ", fp); json_string(fp, r->targets[j]); }
        fprintf(fp, "]}%s\n", i + 1 == e->cxl_region_count ? "" : ",");
    }
    fprintf(fp, "    ]\n  },\n");

    fprintf(fp, "  \"accounting_semantics\": {\n");
    fprintf(fp, "    \"bandwidth_Bps\": \"full cache-line logical bytes explicitly loaded/stored by benchmark kernel divided by measured elapsed time\",\n");
    fprintf(fp, "    \"normal_write_note\": \"RFO and writeback physical traffic are not inferred from logical bytes; use PMU/fabric counters for physical-link traffic\",\n");
    fprintf(fp, "    \"latency_ns\": \"elapsed wall-clock nanoseconds per hop of one dependent randomized pointer chain under simultaneous bandwidth load\"\n  },\n");

    fprintf(fp, "  \"raw_results\": [\n");
    for (size_t i = 0; i < s->raw_count; i++) {
        struct raw_result *r = &s->raw[i];
        bool post_ok = s->residency_post.query_supported && s->residency_post.other_node == 0 && s->residency_post.errors == 0 && s->residency_post.on_target == s->residency_post.samples;
        bool final_valid = r->valid && post_ok;
        char reason[768]; snprintf(reason, sizeof(reason), "%s", r->invalid_reason);
        if (!post_ok) append_reason(reason, sizeof(reason), "post-suite NUMA residency is not 100% verified on target node");
        fprintf(fp, "    {\"pattern\": "); json_string(fp, tp_names[r->pattern]);
        fprintf(fp, ", \"threads\": %d, \"repeat\": %d, \"read_bytes\": %llu, \"write_bytes\": %llu,",
                r->threads, r->repeat, (unsigned long long)r->read_bytes, (unsigned long long)r->write_bytes);
        fprintf(fp, " \"elapsed_sec\": %.9f, \"bandwidth_Bps\": %.3f, \"bandwidth_GBps\": %.6f, \"bandwidth_GiBps\": %.6f,",
                r->elapsed_sec, r->bandwidth_Bps, r->bandwidth_Bps / 1e9, r->bandwidth_Bps / (double)GIB);
        fprintf(fp, " \"latency_ns\": %.6f, \"latency_hops\": %llu,",
                r->latency_ns, (unsigned long long)r->latency_hops);
        fprintf(fp, " \"minor_faults\": %ld, \"major_faults\": %ld, \"voluntary_cs\": %ld, \"involuntary_cs\": %ld,",
                r->minor_faults, r->major_faults, r->voluntary_cs, r->involuntary_cs);
        fprintf(fp, " \"affinity_verified\": %s, \"timing_valid\": %s, \"valid\": %s, \"invalid_reason\": ",
                r->affinity_verified ? "true" : "false", r->timing_valid ? "true" : "false", final_valid ? "true" : "false");
        json_string(fp, reason);
        fprintf(fp, "}%s\n", i + 1 == s->raw_count ? "" : ",");
    }
    fprintf(fp, "  ],\n");

    fprintf(fp, "  \"summary\": {\n");
    for (int p = 0; p < NUM_PATTERNS; p++) {
        fprintf(fp, "    "); json_string(fp, tp_names[p]); fprintf(fp, ": [\n");
        for (int t = 1; t <= s->opt.max_threads; t++) {
            struct summary_result z = summarize(s, (traffic_pattern_t)p, t);
            fprintf(fp, "      {\"threads\": %d, \"runs\": %d, \"valid_runs\": %d, ", t, z.n_total, z.n_valid);
            if (z.n_valid > 0) {
                fprintf(fp, "\"bandwidth_median_Bps\": %.3f, \"bandwidth_min_Bps\": %.3f, \"bandwidth_max_Bps\": %.3f, \"bandwidth_cv_pct\": %.6f, ", z.bw_median_Bps, z.bw_min_Bps, z.bw_max_Bps, z.bw_cv_pct);
                fprintf(fp, "\"latency_median_ns\": %.6f, \"latency_min_ns\": %.6f, \"latency_max_ns\": %.6f, \"latency_cv_pct\": %.6f", z.lat_median_ns, z.lat_min_ns, z.lat_max_ns, z.lat_cv_pct);
            } else {
                fprintf(fp, "\"bandwidth_median_Bps\": null, \"bandwidth_min_Bps\": null, \"bandwidth_max_Bps\": null, \"bandwidth_cv_pct\": null, \"latency_median_ns\": null, \"latency_min_ns\": null, \"latency_max_ns\": null, \"latency_cv_pct\": null");
            }
            fprintf(fp, "}%s\n", t == s->opt.max_threads ? "" : ",");
        }
        fprintf(fp, "    ]%s\n", p + 1 == NUM_PATTERNS ? "" : ",");
    }
    fprintf(fp, "  }\n}\n");

    if (fflush(fp) != 0) die("fflush report failed");
    if (fsync(fileno(fp)) != 0) die("fsync report failed");
    if (fclose(fp) != 0) die("fclose report failed");
    if (rename(tmp, final) != 0) die("atomic rename report failed: %s", strerror(errno));
    printf("Report: %s\n", final);
}

/* ---------- Allocation ---------- */
static void allocate_memory(struct benchmark_state *s)
{
    s->alloc_size = (size_t)(s->opt.total_gib * (double)GIB);
    s->lat_size = (size_t)(s->opt.latency_gib * (double)GIB);
    s->alloc_size &= ~(size_t)(CACHE_LINE_SIZE - 1U);
    s->lat_size &= ~(size_t)(CACHE_LINE_SIZE - 1U);
    if (s->lat_size < 64U * MIB || s->lat_size >= s->alloc_size)
        die("latency working set must be >=64MiB and < total allocation");
    s->bw_size = s->alloc_size - s->lat_size;

    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (s->opt.page_mode == PAGE_HUGETLB) {
        flags |= MAP_HUGETLB;
        if (s->opt.huge_shift == 21) flags |= MAP_HUGE_2MB;
        else if (s->opt.huge_shift == 30) flags |= MAP_HUGE_1GB;
    }
    s->mem = mmap(NULL, s->alloc_size, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (s->mem == MAP_FAILED)
        die("mmap(%s, %.3f GiB) failed: %s", page_mode_name(s->opt.page_mode), (double)s->alloc_size/(double)GIB, strerror(errno));
    if (s->opt.page_mode == PAGE_THP) {
        if (madvise(s->mem, s->alloc_size, MADV_HUGEPAGE) != 0)
            die("MADV_HUGEPAGE failed in strict THP mode: %s", strerror(errno));
    } else if (s->opt.page_mode == PAGE_BASE) {
#ifdef MADV_NOHUGEPAGE
        if (madvise(s->mem, s->alloc_size, MADV_NOHUGEPAGE) != 0)
            warnx("MADV_NOHUGEPAGE failed: %s", strerror(errno));
#endif
    }
    if (bind_memory_to_node(s->mem, s->alloc_size, s->opt.cxl_node) != 0)
        die("mbind(MPOL_BIND|MPOL_MF_STRICT) to node %d failed: %s", s->opt.cxl_node, strerror(errno));

    /* Prefault every byte and initialize predictable contents before building chase. */
    memset(s->mem, 0xA5, s->alloc_size);
    __asm__ volatile("" ::: "memory");
    s->mapping = inspect_mapping(s->mem);
    if (s->opt.page_mode == PAGE_THP) {
        uint64_t expected_kb = (uint64_t)(s->alloc_size / KIB);
        if (s->mapping.anon_huge_kb != expected_kb)
            die("strict THP mode requested but AnonHugePages=%llu kB, expected=%llu kB",
                (unsigned long long)s->mapping.anon_huge_kb, (unsigned long long)expected_kb);
    }
    if (s->opt.page_mode == PAGE_BASE && s->mapping.anon_huge_kb != 0)
        die("base-page mode requested but mapping still contains %llu kB of AnonHugePages",
            (unsigned long long)s->mapping.anon_huge_kb);
    if (s->opt.page_mode == PAGE_HUGETLB && s->mapping.kernel_page_kb <= (uint64_t)(sysconf(_SC_PAGESIZE) / 1024))
        die("HugeTLB mode requested but smaps did not report a huge KernelPageSize");
    if ((s->opt.page_mode == PAGE_HUGETLB || s->opt.page_mode == PAGE_THP) && s->mapping.kernel_page_kb) {
        uint64_t pg = s->mapping.kernel_page_kb * KIB;
        if ((uint64_t)s->lat_size % pg != 0)
            die("latency working-set split is not aligned to actual huge page size (%llu bytes)", (unsigned long long)pg);
    }
    s->residency_pre = verify_residency(s->mem, s->alloc_size, s->opt.cxl_node, s->opt.residency_samples);
    if (!s->residency_pre.query_supported)
        die("cannot verify NUMA page residency with move_pages(); strict v1.1 refuses to run");
    if (s->residency_pre.other_node || s->residency_pre.errors || s->residency_pre.on_target != s->residency_pre.samples)
        die("NUMA residency verification failed: target=%zu/%zu other=%zu errors=%zu",
            s->residency_pre.on_target, s->residency_pre.samples, s->residency_pre.other_node, s->residency_pre.errors);

    s->lat_start = build_single_chase(s->mem, s->lat_size, s->opt.seed);
    s->bw_base = (uint8_t *)s->mem + s->lat_size;
}

/* ---------- CLI ---------- */
static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <CPU_NUMA_NODE> <CXL_NUMA_NODE> <MAX_THREADS> [options]\n"
        "Options:\n"
        "  --total-gib N          total benchmark mapping (default %.1f)\n"
        "  --latency-gib N        single-chain latency working set (default %.1f)\n"
        "  --duration-sec N       measurement duration/run (default %.1f)\n"
        "  --warmup-sec N         warmup duration before every recorded run (default %.1f)\n"
        "  --repeats N            recorded runs per point (default %d)\n"
        "  --page-mode MODE       hugetlb|thp|base (default hugetlb; no fallback)\n"
        "  --huge-size SIZE       default|2M|1G (only with hugetlb)\n"
        "  --seed N               deterministic pointer permutation seed\n"
        "  --residency-samples N  sampled pages for pre/post NUMA verification (default %d)\n"
        "  --output DIR           result root directory (default ./cxl-bench-results)\n"
        "\n"
        "v1.0 results are intentionally NOT baseline-compatible with v1.1.\n",
        prog, DEFAULT_TOTAL_GIB, DEFAULT_LAT_GIB, DEFAULT_DURATION_SEC,
        DEFAULT_WARMUP_SEC, DEFAULT_REPEATS, DEFAULT_RESIDENCY_SAMPLES);
}

static void parse_options(int argc, char **argv, struct options *o)
{
    memset(o, 0, sizeof(*o));
    o->total_gib = DEFAULT_TOTAL_GIB;
    o->latency_gib = DEFAULT_LAT_GIB;
    o->duration_sec = DEFAULT_DURATION_SEC;
    o->warmup_sec = DEFAULT_WARMUP_SEC;
    o->repeats = DEFAULT_REPEATS;
    o->page_mode = PAGE_HUGETLB;
    o->residency_samples = DEFAULT_RESIDENCY_SAMPLES;
    snprintf(o->output_dir, sizeof(o->output_dir), "./cxl-bench-results");
    if (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h"))) { usage(argv[0]); exit(0); }
    if (argc < 4) { usage(argv[0]); exit(EXIT_FAILURE); }
    long v;
    if (parse_long_strict(argv[1], 0, INT_MAX, &v)) die("invalid CPU_NUMA_NODE");
    o->cpu_node = (int)v;
    if (parse_long_strict(argv[2], 0, INT_MAX, &v)) die("invalid CXL_NUMA_NODE");
    o->cxl_node = (int)v;
    if (parse_long_strict(argv[3], 1, MAX_THREADS, &v)) die("invalid MAX_THREADS (1..%d)", MAX_THREADS);
    o->max_threads = (int)v;

    for (int i = 4; i < argc; i++) {
        const char *a = argv[i];
#define NEED_VALUE() do { if (++i >= argc) die("%s requires a value", a); } while(0)
        if (!strcmp(a, "--total-gib")) { NEED_VALUE(); if (parse_double_strict(argv[i], 0.125, 1024.0, &o->total_gib)) die("invalid --total-gib"); }
        else if (!strcmp(a, "--latency-gib")) { NEED_VALUE(); if (parse_double_strict(argv[i], 0.0625, 1024.0, &o->latency_gib)) die("invalid --latency-gib"); }
        else if (!strcmp(a, "--duration-sec")) { NEED_VALUE(); if (parse_double_strict(argv[i], 0.1, 3600.0, &o->duration_sec)) die("invalid --duration-sec"); }
        else if (!strcmp(a, "--warmup-sec")) { NEED_VALUE(); if (parse_double_strict(argv[i], 0.0, 3600.0, &o->warmup_sec)) die("invalid --warmup-sec"); }
        else if (!strcmp(a, "--repeats")) { NEED_VALUE(); if (parse_long_strict(argv[i], 1, 100, &v)) die("invalid --repeats"); o->repeats = (int)v; }
        else if (!strcmp(a, "--page-mode")) {
            NEED_VALUE();
            if (!strcmp(argv[i], "hugetlb")) o->page_mode = PAGE_HUGETLB;
            else if (!strcmp(argv[i], "thp")) o->page_mode = PAGE_THP;
            else if (!strcmp(argv[i], "base")) o->page_mode = PAGE_BASE;
            else die("invalid --page-mode: %s", argv[i]);
        }
        else if (!strcmp(a, "--huge-size")) {
            NEED_VALUE();
            if (!strcmp(argv[i], "default")) o->huge_shift = 0;
            else if (!strcasecmp(argv[i], "2M")) o->huge_shift = 21;
            else if (!strcasecmp(argv[i], "1G")) o->huge_shift = 30;
            else die("invalid --huge-size");
        }
        else if (!strcmp(a, "--seed")) {
            NEED_VALUE(); char *end = NULL; errno = 0; unsigned long long x = strtoull(argv[i], &end, 0);
            if (errno || end == argv[i] || *end) die("invalid --seed");
            o->seed = (uint64_t)x;
            o->seed_explicit = true;
        }
        else if (!strcmp(a, "--residency-samples")) { NEED_VALUE(); if (parse_long_strict(argv[i], 64, 1000000, &v)) die("invalid --residency-samples"); o->residency_samples = (size_t)v; }
        else if (!strcmp(a, "--output")) { NEED_VALUE(); if (strlen(argv[i]) >= sizeof(o->output_dir)) die("output path too long"); strcpy(o->output_dir, argv[i]); }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); exit(0); }
        else die("unknown option: %s", a);
#undef NEED_VALUE
    }
    if (o->latency_gib >= o->total_gib) die("--latency-gib must be less than --total-gib");
    if (!o->seed_explicit) {
        struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
        o->seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ (uint64_t)getpid();
    }
}

static void setup_cpus(struct benchmark_state *s)
{
    s->cpu_count = node_cpu_choices(s->opt.cpu_node, s->cpus, MAX_CPUS, s->env.cpu_cpulist, sizeof(s->env.cpu_cpulist));
    if (s->cpu_count < 2) die("CPU node %d has fewer than 2 usable CPUs in current affinity/cpuset", s->opt.cpu_node);
    s->latency_cpu = s->cpus[0].cpu; /* reserve one physical-core-first CPU */
    s->bw_cpu_count = 0;
    for (int i = 1; i < s->cpu_count && s->bw_cpu_count < MAX_THREADS; i++)
        s->bw_cpus[s->bw_cpu_count++] = s->cpus[i].cpu;
    if (s->opt.max_threads > s->bw_cpu_count) {
        warnx("MAX_THREADS capped from %d to %d because one CPU is reserved for latency and cpuset permits only %d others",
              s->opt.max_threads, s->bw_cpu_count, s->bw_cpu_count);
        s->opt.max_threads = s->bw_cpu_count;
    }
    s->env.llc_bytes = get_llc_bytes(s->latency_cpu);
    char p[PATH_MAX];
    snprintf(p, sizeof(p), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", s->latency_cpu);
    read_text_file(p, s->env.cpu_governor, sizeof(s->env.cpu_governor));
}

static void preflight(struct benchmark_state *s)
{
    if (!s->env.cxl_sysfs_present)
        die("/sys/bus/cxl/devices is absent; strict v1.1 refuses to label this as a CXL benchmark");
    if (!s->env.cxl_node_overlaps_region && !s->env.cxl_node_has_memdev)
        die("target NUMA node %d cannot be correlated to a CXL region HPA range or CXL memdev; strict v1.1 refuses to run", s->opt.cxl_node);
    if (!s->env.cxl_node_overlaps_region)
        warnx("target node does not overlap any visible CXL region HPA range; only memdev NUMA-node correlation is available");
    uint64_t bw_bytes = (uint64_t)((s->opt.total_gib - s->opt.latency_gib) * (double)GIB);
    if (s->env.llc_bytes && bw_bytes < 4ULL * s->env.llc_bytes)
        die("bandwidth working set (%llu bytes) is <4x detected LLC (%llu bytes); enlarge --total-gib",
            (unsigned long long)bw_bytes, (unsigned long long)s->env.llc_bytes);
    if (s->env.numa_balancing > 0)
        warnx("automatic NUMA balancing is enabled; strict residency is verified before and after the suite, but disabling it is recommended for reproducibility");
    if (!s->env.cxl_identity_proven_single_device)
        warnx("target node is not proven to be a single-memdev, one-way CXL region; report scope remains aggregate/end-to-end, not 'device DRAM physical maximum'");
}

static void print_header(struct benchmark_state *s)
{
    printf("CXL Physical Benchmark Engine v%s (schema %s)\n", VERSION, SCHEMA_VERSION);
    printf("IMPORTANT: v1.0 results are NOT baseline-compatible.\n");
    printf("Host=%s Kernel=%s CPU=%s\n", s->env.hostname, s->env.kernel_release, s->env.cpu_model);
    printf("CPU node=%d, CXL node=%d, latency CPU=%d, BW CPUs=%d, scope=%s\n",
           s->opt.cpu_node, s->opt.cxl_node, s->latency_cpu, s->bw_cpu_count, s->env.measurement_scope);
    printf("Page mode=%s, allocation=%.3f GiB (lat=%.3f GiB, bw=%.3f GiB), seed=%llu\n",
           page_mode_name(s->opt.page_mode), (double)s->alloc_size/(double)GIB,
           (double)s->lat_size/(double)GIB, (double)s->bw_size/(double)GIB,
           (unsigned long long)s->opt.seed);
    printf("Pre-residency: %.3f%% on target (%zu/%zu)\n",
           s->residency_pre.target_pct, s->residency_pre.on_target, s->residency_pre.samples);
}

int main(int argc, char **argv)
{
    struct benchmark_state s;
    memset(&s, 0, sizeof(s));
    parse_options(argc, argv, &s.opt);
    gather_environment(&s);
    setup_cpus(&s);
    preflight(&s);
    allocate_memory(&s);
    print_header(&s);

    for (int p = 0; p < NUM_PATTERNS; p++) {
        printf("\nPattern %s\n", tp_names[p]);
        for (int t = 1; t <= s.opt.max_threads; t++) {
            printf("  threads=%d\n", t);
            for (int r = 1; r <= s.opt.repeats; r++) {
                if (s.opt.warmup_sec > 0.0)
                    (void)run_once(&s, (traffic_pattern_t)p, t, 0, s.opt.warmup_sec, false);
                struct raw_result x = run_once(&s, (traffic_pattern_t)p, t, r, s.opt.duration_sec, true);
                printf("    run=%d BW=%8.3f GB/s (%8.3f GiB/s) loaded-lat=%8.2f ns faults=%ld/%ld valid=%s%s%s\n",
                       r, x.bandwidth_Bps/1e9, x.bandwidth_Bps/(double)GIB, x.latency_ns,
                       x.minor_faults, x.major_faults, x.valid ? "yes" : "no",
                       x.invalid_reason[0] ? " reason=" : "", x.invalid_reason);
            }
        }
    }

    s.residency_post = verify_residency(s.mem, s.alloc_size, s.opt.cxl_node, s.opt.residency_samples);
    if (!s.residency_post.query_supported || s.residency_post.other_node || s.residency_post.errors ||
        s.residency_post.on_target != s.residency_post.samples) {
        warnx("post-suite residency verification failed; all summary points will have zero valid runs");
    }

    char run_dir[PATH_MAX];
    char compact[32];
    time_t now = time(NULL); struct tm tm; gmtime_r(&now, &tm);
    strftime(compact, sizeof(compact), "%Y%m%dT%H%M%SZ", &tm);
    int n = snprintf(run_dir, sizeof(run_dir), "%s/%s-%s-n%d-pid%d", s.opt.output_dir, compact,
                     s.env.hostname[0] ? s.env.hostname : "host", s.opt.cxl_node, (int)getpid());
    if (n < 0 || (size_t)n >= sizeof(run_dir)) die("run directory path too long");
    if (mkdir_p(run_dir) != 0) die("mkdir_p(%s): %s", run_dir, strerror(errno));
    write_report(&s, run_dir);

    munmap(s.mem, s.alloc_size);
    free(s.raw);
    return 0;
}

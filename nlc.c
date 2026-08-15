#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <x86intrin.h>
#include <numa.h>
#include <numaif.h>

#define DEFAULT_BUFFER_SIZE (400 * 1024 * 1024) // 400MB
#define ITERATIONS          5000000             // 反復回数

// 512B アラインメント構造体（512B までのアクセスブロックを管理）
typedef struct Node {
    struct Node* next_line;
    uint8_t pad[64 - sizeof(struct Node*)];
} __attribute__((aligned(512))) Node;

// CLOCK_MONOTONIC_RAW による TSC 周波数校正
double get_tsc_frequency_hz(void) {
    struct timespec start_ts, end_ts;
    uint32_t ui;
    struct timespec req = { .tv_sec = 0, .tv_nsec = 200 * 1000 * 1000 };

    clock_gettime(CLOCK_MONOTONIC_RAW, &start_ts);
    uint64_t start_tsc = __rdtscp(&ui);

    nanosleep(&req, NULL);

    uint64_t end_tsc = __rdtscp(&ui);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end_ts);

    uint64_t tsc_diff = end_tsc - start_tsc;
    double time_diff_sec = (end_ts.tv_sec - start_ts.tv_sec) + 
                           (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;

    return (double)tsc_diff / time_diff_sec;
}

// 指定アクセスサイズ (target_bytes) に基づいてリングチェーンを生成
void init_pointer_chain(uint8_t* raw_buffer, size_t count, size_t target_bytes) {
    size_t* indices = (size_t*)malloc(count * sizeof(size_t));
    for (size_t i = 0; i < count; i++) indices[i] = i;

    // Fisher-Yates シャッフル
    for (size_t i = count - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }

    size_t num_lines = (target_bytes <= 64) ? 1 : (target_bytes / 64);

    for (size_t i = 0; i < count; i++) {
        size_t curr_idx = indices[i];
        size_t next_idx = indices[(i + 1) % count];

        uint8_t* block_start = raw_buffer + (curr_idx * 512);
        uint8_t* next_block_start = raw_buffer + (next_idx * 512);

        // ブロック内部の 64B ライン同士をチェーン
        for (size_t l = 0; l < num_lines - 1; l++) {
            Node* current_line = (Node*)(block_start + (l * 64));
            current_line->next_line = (Node*)(block_start + ((l + 1) * 64));
        }

        // 最後のラインから次のランダムブロックの先頭へ
        Node* last_line = (Node*)(block_start + ((num_lines - 1) * 64));
        last_line->next_line = (Node*)next_block_start;
    }

    free(indices);
}

// レイテンシ測定実行
void measure_latency(uint8_t* raw_buffer, size_t count, size_t target_bytes, double tsc_freq_hz) {
    init_pointer_chain(raw_buffer, count, target_bytes);

    size_t num_lines = (target_bytes <= 64) ? 1 : (target_bytes / 64);
    Node* curr = (Node*)raw_buffer;

    // ウォームアップ
    for (size_t i = 0; i < count * num_lines; i++) {
        curr = curr->next_line;
    }

    uint32_t ui;
    _mm_mfence();
    uint64_t start_tsc = __rdtscp(&ui);
    _mm_lfence();

    // 64B未満の命令サイズ切り分け
    if (target_bytes < 64) {
        for (size_t i = 0; i < ITERATIONS; i++) {
            switch (target_bytes) {
                case 1:  __asm__ __volatile__("movb (%0), %b0" : "+r"(curr) :: "memory"); break;
                case 4:  __asm__ __volatile__("movl (%0), %e0" : "+r"(curr) :: "memory"); break;
                case 8:  
                case 16: 
                case 32: __asm__ __volatile__("movq (%0), %0"  : "+r"(curr) :: "memory"); break;
            }
        }
    } 
    // 64B以上のアクセス（キャッシュラインを順にポインタチェイス）
    else {
        for (size_t i = 0; i < ITERATIONS; i++) {
            for (size_t l = 0; l < num_lines; l++) {
                __asm__ __volatile__("movq (%0), %0" : "+r"(curr) :: "memory");
            }
        }
    }

    _mm_lfence();
    uint64_t end_tsc = __rdtscp(&ui);
    _mm_mfence();

    uint64_t total_cycles = end_tsc - start_tsc;
    double total_time_ns = ((double)total_cycles / tsc_freq_hz) * 1e9;

    double ns_per_block = total_time_ns / ITERATIONS;
    double cycles_per_block = (double)total_cycles / ITERATIONS;

    printf("| %4zu Bytes | %10.2f cycles | %8.2f ns |\n", 
           target_bytes, cycles_per_block, ns_per_block);
}

int main(int argc, char* argv[]) {
    srand((unsigned int)time(NULL));

    if (numa_available() < 0) {
        fprintf(stderr, "Error: NUMA is not supported on this system.\n");
        return EXIT_FAILURE;
    }

    int target_node = 0;
    if (argc > 1) {
        target_node = atoi(argv[1]);
    }

    int max_node = numa_max_node();
    if (target_node < 0 || target_node > max_node) {
        fprintf(stderr, "Error: Invalid NUMA node %d. Max node is %d.\n", target_node, max_node);
        return EXIT_FAILURE;
    }

    // CXL / Memory-Only Node チェック
    struct bitmask* cpus = numa_allocate_cpumask();
    numa_node_to_cpus(target_node, cpus);
    int cpu_count = numa_bitmask_weight(cpus);
    numa_free_cpumask(cpus);

    printf("=== CXL / NUMA Multi-Access Size Latency Benchmark ===\n");
    printf("Target NUMA Node : %d (CPU Cores: %d)\n", target_node, cpu_count);
    if (cpu_count == 0) {
        printf("-> Memory-Only Node Detected (e.g., CXL Type-3 Device)\n");
    }

    printf("\nCalibrating Invariant TSC Frequency...\n");
    double tsc_freq_hz = get_tsc_frequency_hz();
    printf("TSC Frequency: %.4f GHz\n\n", tsc_freq_hz / 1e9);

    size_t buffer_size = DEFAULT_BUFFER_SIZE;
    size_t count = buffer_size / 512; // 512B 単位のブロック数

    // 厳格な NUMA バインド設定 (MPOL_BIND)
    struct bitmask* nodemask = numa_allocate_nodemask();
    numa_bitmask_setbit(nodemask, target_node);
    numa_set_membind(nodemask);
    numa_set_strict(1);

    // CXL ノード上からメモリ領域を直接確保
    uint8_t* raw_buffer = (uint8_t*)numa_alloc(buffer_size);
    numa_free_nodemask(nodemask);

    if (!raw_buffer) {
        perror("numa_alloc failed (Could not allocate on target CXL node)");
        return EXIT_FAILURE;
    }

    size_t sizes[] = {1, 4, 8, 16, 32, 64, 128, 256, 384, 512};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("+------------+-------------------+------------+\n");
    printf("| Access Size| Latency (Cycles)  | Latency    |\n");
    printf("+------------+-------------------+------------+\n");

    for (size_t i = 0; i < num_sizes; i++) {
        measure_latency(raw_buffer, count, sizes[i], tsc_freq_hz);
    }

    printf("+------------+-------------------+------------+\n");

    numa_free(raw_buffer, buffer_size);
    return EXIT_SUCCESS;
}

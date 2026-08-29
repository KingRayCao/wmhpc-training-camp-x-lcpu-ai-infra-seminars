#include <cuda_fp8.h>
#include <cstdlib>
#include "../common.h"
#include <cstdint>
#include <random>

#define PACK4(p0, p1, p2, p3) ((unsigned)(p0) | (unsigned)(p1) << 8 | (unsigned)(p2) << 16 | (unsigned)(p3) << 24)

__global__ void mma(const uint8_t *A, const uint8_t *B, float *D)
{
    int lane = threadIdx.x;
    int group = lane >> 2;
    int tig = lane & 3;
    int tig4 = tig << 2;

    unsigned ra[4];
    ra[0] = PACK4(A[group * 32 + tig4],
                  A[group * 32 + tig4 + 1],
                  A[group * 32 + tig4 + 2],
                  A[group * 32 + tig4 + 3]);
    ra[1] = PACK4(A[(group + 8) * 32 + tig4],
                  A[(group + 8) * 32 + tig4 + 1],
                  A[(group + 8) * 32 + tig4 + 2],
                  A[(group + 8) * 32 + tig4 + 3]);
    ra[2] = PACK4(A[group * 32 + tig4 + 16],
                  A[group * 32 + tig4 + 17],
                  A[group * 32 + tig4 + 18],
                  A[group * 32 + tig4 + 19]);
    ra[3] = PACK4(A[(group + 8) * 32 + tig4 + 16],
                  A[(group + 8) * 32 + tig4 + 17],
                  A[(group + 8) * 32 + tig4 + 18],
                  A[(group + 8) * 32 + tig4 + 19]);

    unsigned rb[2];
    rb[0] = PACK4(B[(tig4 + 0) * 8 + group],
                  B[(tig4 + 1) * 8 + group],
                  B[(tig4 + 2) * 8 + group],
                  B[(tig4 + 3) * 8 + group]);
    rb[1] = PACK4(B[(tig4 + 16) * 8 + group],
                  B[(tig4 + 17) * 8 + group],
                  B[(tig4 + 18) * 8 + group],
                  B[(tig4 + 19) * 8 + group]);

    float rc[4] = {0.0f, 0.0f, 0.0f, 0.0f}, rd[4];
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(rd[0]), "=f"(rd[1]), "=f"(rd[2]), "=f"(rd[3])
        : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
          "r"(rb[0]), "r"(rb[1]),
          "f"(rc[0]), "f"(rc[1]), "f"(rc[2]), "f"(rc[3]));

    D[group * 8 + tig * 2] = rd[0];
    D[group * 8 + tig * 2 + 1] = rd[1];
    D[(group + 8) * 8 + tig * 2] = rd[2];
    D[(group + 8) * 8 + tig * 2 + 1] = rd[3];
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        return 1;
    }
    const unsigned int seed = static_cast<unsigned int>(std::strtoul(argv[1], nullptr, 10));
    std::srand(seed);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 15);

    uint8_t hA[16 * 32], hB[32 * 8];
    float fA[16 * 32], fB[32 * 8], ref[16 * 8] = {};
    for (int r = 0; r < 16 * 32; r++)
    {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hA[r] = *(uint8_t*)&v;
        fA[r] = float(v);
    }
    for (int r = 0; r < 32 * 8; r++)
    {
        __nv_fp8_e4m3 v = __nv_fp8_e4m3((float)(dist(rng) - 8));
        hB[r] = *(uint8_t*)&v;
        fB[r] = float(v);
    }
    for (int r = 0; r < 16; r++)
    {
        for (int c = 0; c < 8; c++)
        {
            for (int k = 0; k < 32; k++)
            {
                ref[r * 8 + c] += fA[r * 32 + k] * fB[k * 8 + c];
            }
        }
    }

    uint8_t *dA, *dB;
    float *dD;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(hA)));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(hB)));
    CUDA_CHECK(cudaMalloc(&dD, sizeof(ref)));
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice));
    mma<<<1, 32>>>(dA, dB, dD);
    float hD[16 * 8];
    CUDA_CHECK(cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost));

    long bad = 0;
    for (int i = 0; i < 16 * 8; i++)
    {
        if (std::abs(hD[i] - ref[i]) > 1e-3f)
        {
            bad++;
        }
    }
    if (bad)
        printf("FAIL: %ld mismatches\n", bad);
    else
        printf("PASS\n");
    return bad != 0;
}

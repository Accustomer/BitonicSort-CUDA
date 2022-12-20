#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
    }                                                                          \
}



/***** Host functions *****/
inline bool compare(int a, int b, bool descending = true)
{
    return (a < b&& descending) || (a > b && !descending);
}


void mergeSort(int* a, int n, bool descending = true)
{
    int stride = n >> 1;
    int t = 0;
    for (int i = 0, j = stride; i < stride; i++, j++)
    {
        if (compare(a[i], a[j], descending))
        {
            t = a[j];
            a[j] = a[i];
            a[i] = t;
        }
    }
    if (stride >= 2)
    {
        mergeSort(a, stride, descending);
        mergeSort(a + stride, stride, descending);
    }
}


void hBitonicSortRecursive(int* a, int n, bool descending)
{
    int stride = 2;
    int inter_step = 1;
    while (stride <= n)
    {
        inter_step = (stride << 1);
        // Order
        for (int i = 0; i < n; i += inter_step)
        {
            mergeSort(a + i, stride, descending);
        }
        // Reverse order
        for (int i = stride; i < n; i += inter_step)
        {
            mergeSort(a + i, stride, !descending);
        }
        stride = inter_step;
    }
}


void hBitonicSort(int* a, int n, bool descending)
{
    int t = 0;
    int half_stride = 1, hs = 1, s = 2;
    int hn = n >> 1;
    for (int stride = 2; stride <= n; stride <<= 1)
    {
        s = stride;
        while (s >= 2)
        {
            hs = s >> 1;
            for (int i = 0; i < hn; i++)
            {
                bool orange = (i / half_stride) % 2 == 0;
                int j = (i / hs) * s + (i % hs);
                int k = j + hs;
                //printf("Stride: %d, s: %d, i: %d, j: %d, k: %d\n", stride, s, i, j, k);
                if ((descending && ((orange && a[j] < a[k]) || (!orange && a[j] > a[k]))) ||
                    (!descending && ((orange && a[j] > a[k]) || (!orange && a[j] < a[k]))))
                {
                    t = a[k];
                    a[k] = a[j];
                    a[j] = t;
                }
            }
            s = hs;
        }
        half_stride = stride;
    }
}


void checkResult(int* a, int* b, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != b[i])
        {
            printf("Sorting failed!\n");
            return;
        }
    }
    printf("Sorting success!\n");
}


inline long long cpuTimer()
{
    std::chrono::microseconds ms = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
        );
    return ms.count();
}




/***** Device functions *****/
__global__ void gBitonicSort(int* a, int n_p, bool descending)
{
    unsigned int tid = threadIdx.x;

    int stride_p, half_stride_p, s_p, hs_p, hs, i, j, k, t, hn;
    bool orange;
    hn = 1 << (n_p - 1);
    half_stride_p = 0;
    for (stride_p = 1; stride_p <= n_p; stride_p++)
    {
        s_p = stride_p;
        while (s_p >= 1)
        {
            hs_p = s_p - 1;
            hs = 1 << hs_p;
            for (i = tid; i < hn; i += blockDim.x)
            {
                orange = (i >> half_stride_p) % 2 == 0;
                j = ((i >> hs_p) << s_p) + (i % hs);
                k = j + hs;
                if ((descending && ((orange && a[j] < a[k]) || (!orange && a[j] > a[k]))) ||
                    (!descending && ((orange && a[j] > a[k]) || (!orange && a[j] < a[k]))))
                {
                    t = a[k];
                    a[k] = a[j];
                    a[j] = t;
                }
            }
            __syncthreads();
            s_p = hs_p;
        }
        half_stride_p++;
    }
}




int main(int argc, char** argv)
{
    int p = 14;
    int alg = 0;
    bool descending = false;
    if (argc > 1) p = atoi(argv[1]);
    if (argc > 2) alg = atoi(argv[2]);
    if (argc > 3) descending = (bool)atoi(argv[3]);

    // Number of values
    int n = 1 << p;
    const char* alg_str = alg == 0 ? "hBitonicSortRecursive" : (alg == 1 ? "hBitonicSort" : "gBitonicSort");
    const char* order_str = descending ? "descending" : "ascending";
    printf("Argument 0, number of values:    %d\n", n);
    printf("Argument 1, selected algorithms: %s\n", alg_str);
    printf("Argument 2, descending:          %s\n", order_str);

    // Initialize at host side
    size_t nbytes = n * sizeof(int);
    int* hvals = (int*)malloc(nbytes);
    std::vector<int> hvec(n);
    for (int i = 0; i < n; i++)
    {
        hvals[i] = rand() % n;
        hvec[i] = hvals[i];
    }

    // Sort by STL
    long long t_start = 0, t_elapse = 0;
    t_start = cpuTimer();
    std::sort(hvec.begin(), hvec.end());
    if (descending)
    {
        std::reverse(hvec.begin(), hvec.end());
    }
    t_elapse = cpuTimer() - t_start;
    printf("Time cost on STL sort: %fms\n", (float)t_elapse / 1000);

    // Test algorithms
    switch (alg)
    {
    case 0:
        t_start = cpuTimer();
        hBitonicSortRecursive(hvals, n, descending);
        t_elapse = cpuTimer() - t_start;
        break;
    case 1:
        t_start = cpuTimer();
        hBitonicSort(hvals, n, descending);
        t_elapse = cpuTimer() - t_start;
        break;
    case 2:
    {
        int* dvals = NULL;
        CHECK(cudaMalloc((void**)&dvals, nbytes));
        CHECK(cudaMemcpy(dvals, hvals, nbytes, cudaMemcpyHostToDevice));
        t_start = cpuTimer();
        gBitonicSort << <1, 1024 >> > (dvals, p, descending);
        CHECK(cudaDeviceSynchronize());
        t_elapse = cpuTimer() - t_start;
        CHECK(cudaMemcpy(hvals, dvals, nbytes, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(dvals));
        break;
    }
    default:
        printf("Only algorithm 0, 1, 2 were supported!\n");
        break;
    }
    printf("Time cost on %s: %fms\n", alg_str, (float)t_elapse / 1000);
    checkResult(hvals, hvec.data(), n);

    // Free host data
    free(hvals);

    return EXIT_SUCCESS;
}

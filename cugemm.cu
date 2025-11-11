// TODO: before you submit on Canvas, include here:
//   1) which GPU you used and
//   2) what performance improvement you obtained over previous homework(s)

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <cstdint>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// from https://github.com/jarro2783/cxxopts
#include "cxxopts.hpp"

#define cudaCheck(err) (cudaErrorCheck(err, __FILE__, __LINE__))
#define cublasCheck(err) (cublasErrorCheck(err, __FILE__, __LINE__))
#define ROUND_UP_TO_NEAREST(M, N) (((M) + (N) - 1) / (N))

#define uint uint32_t

const uint F = 32;
const uint G = 4;

enum Algo
{
    cublas = 0,
    basic,
    gmem_coalesced,
    smem,
    smem_multioutput,
    cuda_streams,
    numAlgos,
};

const char *algo2str(Algo a)
{
    switch (a)
    {
    case cublas:
        return "cublas";
    case basic:
        return "basic";
    case gmem_coalesced:
        return "gmem_coalesced";
    case smem:
        return "sharedmem";
    case smem_multioutput:
        return "sharedmem_multioutput";
    case cuda_streams:
        return "cuda_streams";
    default:
        return "INVALID";
    }
}

void cudaErrorCheck(cudaError_t error, const char *file, int line);
void cublasErrorCheck(cublasStatus_t status, const char *file, int line);
void randomize_matrix(float *mat, int N);
void const_init_matrix(float *mat, int N, float F);
bool verify_matrix(float *expected, float *actual, int M, int N);
void print_matrix(const float *A, int M, int N, std::ostream &outs);
void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

const std::string errLogFile = "gemmValidationFailure.txt";

// NB: must use a single generator to avoid duplicates
std::default_random_engine generator(2);
std::uniform_real_distribution<float> distribution(0, 1);

void cuda_streams_algo(int M, int N, int K, float alpha,
                       float *A, float *B, float beta, float *C, float *hA, float *hB, float *hC, int streams);

int main(int argc, char **argv)
{
    // command-line flags
    cxxopts::Options options("gemm.cu", "CUDA GEMM kernels");
    options.add_options()("size", "matrix size (N x N)", cxxopts::value<uint16_t>()->default_value("128"))                //
        ("reps", "repeat GEMM this many times", cxxopts::value<uint16_t>()->default_value("1"))                           //
        ("algo", "GEMM algorithm to use, a number in [0,4], 0 is cuBLAS", cxxopts::value<uint16_t>()->default_value("0")) //
        ("validate", "Validate output against cuBLAS", cxxopts::value<bool>()->default_value("true"))                     //
        ("rngseed", "PRNG seed", cxxopts::value<uint>()->default_value("2"))                                              //
        ("streams", "Number of streams for the pipelined memcpy", cxxopts::value<uint>()->default_value("8"))                                              //
        ("h,help", "Print usage");

    auto clFlags = options.parse(argc, argv);
    if (clFlags.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    const uint16_t SIZE = clFlags["size"].as<uint16_t>();
    if (SIZE % 32 != 0)
    {
        // std::cout << "--size must be a multiple of 32" << std::endl;
        // exit(EXIT_FAILURE);
    }
    const uint16_t REPS = clFlags["reps"].as<uint16_t>();
    const Algo ALGO = static_cast<Algo>(clFlags["algo"].as<uint16_t>());
    if (ALGO >= numAlgos)
    {
        printf("Invalid algorithm: %d\n", ALGO);
        exit(EXIT_FAILURE);
    }

    const bool VALIDATE = clFlags["validate"].as<bool>();
    const uint SEED = clFlags["rngseed"].as<uint>();
    const uint STREAMS = clFlags["streams"].as<uint>();
    generator.seed(SEED);
    printf("Multiplying two %u x %u matrices with %u trials using %s algorithm\n", SIZE, SIZE, REPS, algo2str(ALGO));

    cudaCheck(cudaSetDevice(0));

    // Setup cublas
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaCheck(cudaEventCreate(&beg));
    cudaCheck(cudaEventCreate(&end));

    uint16_t m = SIZE, n = SIZE, k = SIZE;

    // GEMM computes C = alpha*AB+beta*C

    // just do pure A*B (for simpler debugging)
    float alpha = 1.0, beta = 1.0, initC = 1.0;

    float *A = nullptr, *B = nullptr, *C = nullptr, *C_ref = nullptr;     // host matrices
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dC_ref = nullptr; // device matrices

    A = (float *)malloc(sizeof(float) * SIZE * SIZE);
    B = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C = (float *)malloc(sizeof(float) * SIZE * SIZE);
    C_ref = (float *)malloc(sizeof(float) * SIZE * SIZE);

    randomize_matrix(A, SIZE * SIZE);
    randomize_matrix(B, SIZE * SIZE);
    randomize_matrix(C, SIZE * SIZE);

    const_init_matrix(C, SIZE * SIZE, initC);
    // print_matrix(A, SIZE, SIZE, std::cout);
    // print_matrix(B, SIZE, SIZE, std::cout);
    // print_matrix(C, SIZE, SIZE, std::cout);

    printf("dimensions(m=n=k) %u, alpha: %f, beta: %f\n", m, alpha, beta);

    cudaCheck(cudaMalloc((void **)&dA, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dB, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC, sizeof(float) * SIZE * SIZE));
    cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * SIZE * SIZE));

    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));

    if (ALGO != cuda_streams)
    {
        cudaCheck(cudaMemcpy(dA, A, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dB, B, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dC, C, sizeof(float) * SIZE * SIZE, cudaMemcpyHostToDevice));
    }
    else
    {
        // do a piplined memcpy of dA, dB, dC
        cuda_streams_algo(m, n, k, alpha, dA, dB, beta, dC, A, B, C, int(STREAMS));
    }

    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (!VALIDATE)
    {
        printf("disabled validation\n");
    }
    else
    {
        // run cublas to get correct answer in dC_ref
        runCublas(handle, m, n, k, alpha, dA, dB, beta, dC_ref);

        // run user's algorithm, filling in dC
        runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);

        cudaCheck(cudaDeviceSynchronize());

        // copy both results back to host
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (verify_matrix(C_ref, C, n, m))
        {
            printf("Validated successfully!\n");
        }
        else
        {
            printf("Failed validation against NVIDIA cuBLAS.\n");
            std::cout << " Logging faulty output into " << errLogFile << "\n";
            std::ofstream fs;
            fs.open(errLogFile, std::ios::out | std::ios::trunc);
            fs << "alpha=" << alpha << " beta=" << beta << std::endl;
            fs << "C matrix initialized to " << initC << std::endl
               << std::endl;
            fs << "A:" << std::endl;
            print_matrix(A, m, n, fs);
            fs << "B:" << std::endl;
            print_matrix(B, m, n, fs);
            fs << "C:" << std::endl;
            print_matrix(C, m, n, fs);
            fs << "Expected:" << std::endl;
            print_matrix(C_ref, m, n, fs);
            fs.close();
            exit(EXIT_FAILURE);
        }
    }

    // timing run(s)
    cudaEventRecord(beg);
    for (int j = 0; j < REPS; j++)
    {
        // We don't reset dC between runs to save time
        if ( ALGO == cuda_streams ) {
            cuda_streams_algo(m, n, k, alpha, dA, dB, beta, dC, A, B, C, int(STREAMS));
        }
        else {
            runAlgo(ALGO, handle, m, n, k, alpha, dA, dB, beta, dC);
        }
        cudaCheck(cudaDeviceSynchronize());
    }

    // TODO: measure timing without memory transfers?
    cudaCheck(cudaEventRecord(end));
    cudaCheck(cudaEventSynchronize(beg));
    cudaCheck(cudaEventSynchronize(end));
    float elapsed_time;
    cudaCheck(cudaEventElapsedTime(&elapsed_time, beg, end));
    elapsed_time /= 1000.; // Convert to seconds

    double flops = (double)2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.2f) GFLOPS. size: (%u).\n",
        elapsed_time / REPS,
        (REPS * flops * 1e-9) / elapsed_time,
        m);

    // free CPU and GPU memory
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaCheck(cudaFree(dA));
    cudaCheck(cudaFree(dB));
    cudaCheck(cudaFree(dC));
    cudaCheck(cudaFree(dC_ref));
    cublasCheck(cublasDestroy(handle));

    return 0;
}

__global__ void runSharedMemMultiOutput(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);

void cuda_streams_algo(int M, int N, int K, float alpha,
                       float *A, float *B, float beta, float *C, float *hA, float *hB, float *hC, int num_streams)
{
    printf("CUDA Streams Algo: %dx%d, %dx%d\n", M, K, K, N);

    // first copy all of B
    cudaCheck(cudaMemcpy(B, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice));

    std::vector<cudaStream_t> streams(num_streams);
    int rows_per_stream = M / num_streams;

    int row_start = 0;
    for (int pass = 0; pass < num_streams; pass++)
    {
        int this_rows = min(rows_per_stream, M - row_start);

        float* offsetA = A + row_start * K;
        float* offsetC = C + row_start * N;
        float* offsetHA = hA + row_start * K;
        float* offsetHC = hC + row_start * N;

        cudaCheck(cudaStreamCreate(&streams[pass]));
        cudaCheck(cudaMemcpyAsync(offsetA, offsetHA, sizeof(float) * this_rows * K, cudaMemcpyHostToDevice, streams[pass]));
        cudaCheck(cudaMemcpyAsync(offsetC, offsetHC, sizeof(float) * this_rows * N, cudaMemcpyHostToDevice, streams[pass]));

        //run the kernel
        dim3 blockSize(F / G, F / G);
        dim3 gridSize(ROUND_UP_TO_NEAREST(N, F), ROUND_UP_TO_NEAREST(this_rows, F));
        runSharedMemMultiOutput<<<gridSize, blockSize, 0, streams[pass]>>>(this_rows, N, K, alpha, offsetA, B, beta, offsetC);

        cudaCheck(cudaMemcpyAsync(offsetHC, offsetC, sizeof(float) * this_rows * N, cudaMemcpyDeviceToHost, streams[pass]));
        row_start += this_rows;
    }

    for (int pass = 0; pass < num_streams; pass++) {
        cudaCheck(cudaStreamSynchronize(streams[pass]));
        cudaCheck(cudaStreamDestroy(streams[pass]));
    }
}

/** Function to check for errors in CUDA API calls */
void cudaErrorCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s: %s\n", file, line,
               cudaGetErrorName(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void cublasErrorCheck(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUDA ERROR] at file %s:%d:\n %s: %s\n", file, line,
               cublasGetStatusName(status), cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

/** Initialize the given matrix `mat` which has `N` contiguous values. Contents of `mat` are set to random values. */
void randomize_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = distribution(generator);
    }
}

void const_init_matrix(float *mat, int N, float F)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = F;
    }
}

/** Print the given MxN matrix `mat` to the provided output stream. */
void print_matrix(const float *A, int M, int N, std::ostream &outs)
{
    outs << "[";
    for (int i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
        {
            outs << std::fixed << std::setprecision(3) << A[i];
        }
        else
        {
            outs << std::fixed << std::setprecision(3) << A[i] << ", ";
        }
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                outs << ";" << std::endl;
        }
    }
    outs << "]" << std::endl
         << std::endl;
}

bool verify_matrix(float *expected, float *actual, int M, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            float fexp = (expected[(i * N) + j]);
            float fact = (actual[(i * N) + j]);
            double diff = std::fabs(fexp - fact);
            if (diff > 0.002)
            {
                printf("Divergence! Should be %5.3f, is %5.3f (diff %5.3f) at [%d,%d]\n",
                       fexp, fact, diff, i, j);
                return false;
            }
        }
    }
    return true;
}

void runCublas(cublasHandle_t handle, int M, int N, int K, float alpha,
               float *A, float *B, float beta, float *C)
{
    // cuBLAS uses *column-major* order. So we change the order of our row-major A &
    // B, since (B^T*A^T)^T = (A*B)
    // cublasStatus_t ok = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16F,
    //                                  N, A, CUDA_R_16F, K, &beta, C, CUDA_R_16F, N, /*CUBLAS_COMPUTE_16F*/ CUBLAS_COMPUTE_16F_PEDANTIC,
    //                                  CUBLAS_GEMM_DEFAULT);
    cublasStatus_t ok = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    cublasCheck(ok);
}

__global__ void runBasic(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0;
        // C = alpha*(AxB)+beta*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(x * K) + i] * B[(i * N) + y];
        }
        // __C__[x][y]
        C[(x * N) + y] = (alpha * tmp) + (beta * C[x * N + y]);
    }
}

__global__ void runGmemCoalesced(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int blocks_per_row = (N + blockDim.x - 1) / blockDim.x;

    int row = blockIdx.x / blocks_per_row;          // which row
    int col_block = blockIdx.x % blocks_per_row;    // which block within the row
    int col = col_block * blockDim.x + threadIdx.x; // which column

    if (row < M && col < N)
    {
        float tmp = 0.0;
        // C = alpha*(AxB)+beta*C
        for (int i = 0; i < K; ++i)
        {
            // tmp += __A__[x][i] * __B__[i][y]
            tmp += A[(row * K) + i] * B[(i * N) + col];
        }
        // __C__[x][y]
        C[(row * N) + col] = (alpha * tmp) + (beta * C[row * N + col]);
    }
}

__global__ void runSharedMem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW2 TODO: Use shared memory to cache square FxF tiles of the A and B matrices in shared memory
    // (SA and SB, respectively, provided below). Each thread should compute the result for one cell
    // of the output matrix C.

    // Note, you will also need to change the grid dimensions in the kernel launch below to take into account the value
    // of F (which is a constant, defined above). You should experiment with different values of F to see how it
    // affects performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    int row = blockRow * F + threadRow;
    int col = blockCol * F + threadCol;

    float tmp = 0.0f;

    for (int t = 0; t < (K + F - 1) / F; ++t)
    {
        if (row < M && t * F + threadCol < K)
        {
            SA[threadRow][threadCol] = A[row * K + t * F + threadCol];
        }
        else
        {
            SA[threadRow][threadCol] = 0.0f;
        }

        if (col < N && t * F + threadRow < K)
        {
            SB[threadRow][threadCol] = B[(t * F + threadRow) * N + col];
        }
        else
        {
            SB[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < F; ++i)
        {
            tmp += SA[threadRow][i] * SB[i][threadCol];
        }

        __syncthreads();
    }

    // Write the final result to C
    if (row < M && col < N)
    {
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

__global__ void runSharedMemMultiOutput(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    // HW3 TODO: Copy your runSharedMem() code here and update it so that each thread computes the result for GxG cells
    // of the output matrix C. Each thread should accumulate temporary results in the local LC matrix, provided below,
    // before writing them to C in global memory.

    // Note, you will also need to change the grid dimensions in the kernel launch below. You should experiment
    // with different values of F and G to see how they affect performance.

    __shared__ float SA[F][F];
    __shared__ float SB[F][F];

    float LC[G][G] = {0.0};

    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int threadRow = threadIdx.y; // 0..F/G
    int threadCol = threadIdx.x; // 0..F/G

    int row = blockRow * F + threadRow * G; // F is a multiple of G, so this is every G
    int col = blockCol * F + threadCol * G;

    for (int t = 0; t < (K + F - 1) / F; ++t)
    {
        for (int gX = 0; gX < G; gX++)
        {
            for (int gY = 0; gY < G; gY++)
            {
                int tempRow = row + gX;
                int tempCol = col + gY;
                int tempThreadRow = G * threadRow + gX;
                int tempThreadCol = G * threadCol + gY;

                if (tempRow < M && t * F + tempThreadCol < K)
                {
                    SA[tempThreadRow][tempThreadCol] = A[tempRow * K + t * F + tempThreadCol];
                }
                else
                {
                    SA[tempThreadRow][tempThreadCol] = 0.0f;
                }

                if (tempCol < N && t * F + tempThreadRow < K)
                {
                    SB[tempThreadRow][tempThreadCol] = B[(t * F + tempThreadRow) * N + tempCol];
                }
                else
                {
                    SB[tempThreadRow][tempThreadCol] = 0.0f;
                }
            }
        }

        __syncthreads();

        for (int gX = 0; gX < G; gX++)
        {
            for (int gY = 0; gY < G; gY++)
            {
                for (int i = 0; i < F; ++i)
                {
                    int tempThreadRow = G * threadRow + gX;
                    int tempThreadCol = G * threadCol + gY;
                    LC[gX][gY] += SA[tempThreadRow][i] * SB[i][tempThreadCol];
                }
            }
        }

        __syncthreads();
    }

    // Write the final result to C
    for (int gX = 0; gX < G; gX++)
    {
        for (int gY = 0; gY < G; gY++)
        {
            int tempRow = row + gX;
            int tempCol = col + gY;
            if (tempRow < M && tempCol < N)
            {
                C[tempRow * N + tempCol] = alpha * LC[gX][gY] + beta * C[tempRow * N + tempCol];
            }
        }
    }
}

void runAlgo(Algo algo, cublasHandle_t handle, int M, int N, int K, float alpha,
             float *A, float *B, float beta, float *C)
{
    switch (algo)
    {
    case cublas:
        runCublas(handle, M, N, K, alpha, A, B, beta, C);
        break;
    case basic:
    {
        dim3 gridSize(ROUND_UP_TO_NEAREST(M, 32), ROUND_UP_TO_NEAREST(N, 32));
        dim3 blockSize(32, 32);
        runBasic<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case gmem_coalesced:
    {
        int blockSize = 128;
        int blocksPerRow = (N + blockSize - 1) / blockSize;
        int gridSize = M * blocksPerRow;
        runGmemCoalesced<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        dim3 blockSize(F, F);
        dim3 gridSize(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
        runSharedMem<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
        break;
    }
    case smem_multioutput:
    {
        assert(0 == M % F);
        assert(0 == N % F);
        assert(0 == K % F);
        assert(0 == F % G);
        assert((F * F) / (G * G) >= F);
        dim3 blockSize(F / G, F / G);
        dim3 gridSize(ROUND_UP_TO_NEAREST(M, F), ROUND_UP_TO_NEAREST(N, F));
        runSharedMemMultiOutput<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
        break;
        break;
    }
    case cuda_streams:
    {
        break;
    }
    default:
        printf("Invalid algorithm: %d\n", algo);
        exit(EXIT_FAILURE);
    }
    cudaCheck(cudaDeviceSynchronize()); // wait for kernel to finish
    cudaCheck(cudaGetLastError());      // check for errors from kernel run
}

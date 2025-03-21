const char* sgemm_desc = "Simple blocked sgemm.";

#include <immintrin.h>
#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 64
#endif
#define BLOCK_SIZE_M 256
#define BLOCK_SIZE_K 128
#define BLOCK_SIZE_N 64
#define SMALLER_BLOCK_SIZE 16
#define min(a,b) (((a)<(b))?(a):(b))

static void Pack_Array (float* Array_old, float* Array_packed, int lda, int lda_new, int FIRST_BLOCK_SIZE, int SECOND_BLOCK_SIZE)
{
  // cannot be changed because if remainder > 0 and just use the AVX-512, maybe there will be a next block will be involved.
  float *array_packed, *array_old;
  array_packed = Array_packed;
  array_old = Array_old;
  int jj,kk;
  for(jj = 0; jj < SECOND_BLOCK_SIZE; jj++)
  {
    for(kk = 0; kk < FIRST_BLOCK_SIZE - 15; kk+=16)
      _mm512_store_ps(&array_packed[kk+jj*lda_new], _mm512_load_ps(&array_old[kk+jj*lda]));

    int remainder = FIRST_BLOCK_SIZE - kk;
    if(remainder > 0)
    {
      __mmask16 mask = ((1 << remainder) - 1);
      __m512 data = _mm512_maskz_load_ps(mask, &array_old[kk+jj*lda]);
      _mm512_mask_store_ps(&array_packed[kk+jj*lda_new], mask, data);
    }
  }
}
static void do_block_avx32x8(int lda, int ldb, int ldc, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  __m512 c00 = _mm512_setzero_ps();
  __m512 c01 = _mm512_setzero_ps();
  __m512 c10 = _mm512_setzero_ps();
  __m512 c11 = _mm512_setzero_ps();
  __m512 c20 = _mm512_setzero_ps();
  __m512 c21 = _mm512_setzero_ps();
  __m512 c30 = _mm512_setzero_ps();
  __m512 c31 = _mm512_setzero_ps();
  __m512 c40 = _mm512_setzero_ps();
  __m512 c41 = _mm512_setzero_ps();
  __m512 c50 = _mm512_setzero_ps();
  __m512 c51 = _mm512_setzero_ps();
  __m512 c60 = _mm512_setzero_ps();
  __m512 c61 = _mm512_setzero_ps();
  __m512 c70 = _mm512_setzero_ps();
  __m512 c71 = _mm512_setzero_ps();
  for (int k = 0; k < K; ++k)
  {
    __m512 a0 = _mm512_load_ps(&A[k*lda]);
    __m512 a1 = _mm512_load_ps(&A[k*lda+16]);
    __m512 b0 = _mm512_set1_ps(B[k+ldb*0]);
    __m512 b1 = _mm512_set1_ps(B[k+ldb*1]);
    __m512 b2 = _mm512_set1_ps(B[k+ldb*2]);
    __m512 b3 = _mm512_set1_ps(B[k+ldb*3]);
    __m512 b4 = _mm512_set1_ps(B[k+ldb*4]);
    __m512 b5 = _mm512_set1_ps(B[k+ldb*5]);
    __m512 b6 = _mm512_set1_ps(B[k+ldb*6]);
    __m512 b7 = _mm512_set1_ps(B[k+ldb*7]);
    c00 = _mm512_fmadd_ps(a0, b0, c00);
    c01 = _mm512_fmadd_ps(a1, b0, c01);
    c10 = _mm512_fmadd_ps(a0, b1, c10);
    c11 = _mm512_fmadd_ps(a1, b1, c11);
    c20 = _mm512_fmadd_ps(a0, b2, c20);
    c21 = _mm512_fmadd_ps(a1, b2, c21);
    c30 = _mm512_fmadd_ps(a0, b3, c30);
    c31 = _mm512_fmadd_ps(a1, b3, c31);
    c40 = _mm512_fmadd_ps(a0, b4, c40);
    c41 = _mm512_fmadd_ps(a1, b4, c41);
    c50 = _mm512_fmadd_ps(a0, b5, c50);
    c51 = _mm512_fmadd_ps(a1, b5, c51);
    c60 = _mm512_fmadd_ps(a0, b6, c60);
    c61 = _mm512_fmadd_ps(a1, b6, c61);
    c70 = _mm512_fmadd_ps(a0, b7, c70);
    c71 = _mm512_fmadd_ps(a1, b7, c71);
  }
  _mm512_store_ps(&C[ldc*0], _mm512_add_ps(_mm512_load_ps(&C[ldc*0]), c00));
  _mm512_store_ps(&C[ldc*0+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*0+16]), c01));
  _mm512_store_ps(&C[ldc*1], _mm512_add_ps(_mm512_load_ps(&C[ldc*1]), c10));
  _mm512_store_ps(&C[ldc*1+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*1+16]), c11));
  _mm512_store_ps(&C[ldc*2], _mm512_add_ps(_mm512_load_ps(&C[ldc*2]), c20));
  _mm512_store_ps(&C[ldc*2+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*2+16]), c21));
  _mm512_store_ps(&C[ldc*3], _mm512_add_ps(_mm512_load_ps(&C[ldc*3]), c30));
  _mm512_store_ps(&C[ldc*3+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*3+16]), c31));
  _mm512_store_ps(&C[ldc*4], _mm512_add_ps(_mm512_load_ps(&C[ldc*4]), c40));
  _mm512_store_ps(&C[ldc*4+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*4+16]), c41));
  _mm512_store_ps(&C[ldc*5], _mm512_add_ps(_mm512_load_ps(&C[ldc*5]), c50));
  _mm512_store_ps(&C[ldc*5+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*5+16]), c51));
  _mm512_store_ps(&C[ldc*6], _mm512_add_ps(_mm512_load_ps(&C[ldc*6]), c60));
  _mm512_store_ps(&C[ldc*6+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*6+16]), c61));
  _mm512_store_ps(&C[ldc*7], _mm512_add_ps(_mm512_load_ps(&C[ldc*7]), c70));
  _mm512_store_ps(&C[ldc*7+16], _mm512_add_ps(_mm512_load_ps(&C[ldc*7+16]), c71));
}
static void do_block_avx64x4(int lda, int ldb, int ldc, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  __m512 c00 = _mm512_setzero_ps();
  __m512 c01 = _mm512_setzero_ps();
  __m512 c02 = _mm512_setzero_ps();
  __m512 c03 = _mm512_setzero_ps();
  __m512 c10 = _mm512_setzero_ps();
  __m512 c11 = _mm512_setzero_ps();
  __m512 c12 = _mm512_setzero_ps();
  __m512 c13 = _mm512_setzero_ps();
  __m512 c20 = _mm512_setzero_ps();
  __m512 c21 = _mm512_setzero_ps();
  __m512 c22 = _mm512_setzero_ps();
  __m512 c23 = _mm512_setzero_ps();
  __m512 c30 = _mm512_setzero_ps();
  __m512 c31 = _mm512_setzero_ps(); 
  __m512 c32 = _mm512_setzero_ps();
  __m512 c33 = _mm512_setzero_ps();
  for(int k = 0; k < K; k++)
  {
    __m512 a0 = _mm512_load_ps(&A[k*lda]);
    __m512 a1 = _mm512_load_ps(&A[k*lda + 16]);
    __m512 a2 = _mm512_load_ps(&A[k*lda + 32]);
    __m512 a3 = _mm512_load_ps(&A[k*lda + 48]);
    __m512 b0 = _mm512_set1_ps(B[k + ldb*0]);
    __m512 b1 = _mm512_set1_ps(B[k + ldb*1]);
    __m512 b2 = _mm512_set1_ps(B[k + ldb*2]);
    __m512 b3 = _mm512_set1_ps(B[k + ldb*3]);
    c00 = _mm512_fmadd_ps(a0, b0, c00);
    c01 = _mm512_fmadd_ps(a0, b1, c01);
    c02 = _mm512_fmadd_ps(a0, b2, c02);
    c03 = _mm512_fmadd_ps(a0, b3, c03);
    c10 = _mm512_fmadd_ps(a1, b0, c10);
    c11 = _mm512_fmadd_ps(a1, b1, c11);
    c12 = _mm512_fmadd_ps(a1, b2, c12);
    c13 = _mm512_fmadd_ps(a1, b3, c13);
    c20 = _mm512_fmadd_ps(a2, b0, c20);
    c21 = _mm512_fmadd_ps(a2, b1, c21);
    c22 = _mm512_fmadd_ps(a2, b2, c22);
    c23 = _mm512_fmadd_ps(a2, b3, c23);
    c30 = _mm512_fmadd_ps(a3, b0, c30);
    c31 = _mm512_fmadd_ps(a3, b1, c31);
    c32 = _mm512_fmadd_ps(a3, b2, c32);
    c33 = _mm512_fmadd_ps(a3, b3, c33);
  }
  _mm512_store_ps(&C[ldc*0], _mm512_add_ps(_mm512_load_ps(&C[ldc*0]), c00));
  _mm512_store_ps(&C[ldc*1], _mm512_add_ps(_mm512_load_ps(&C[ldc*1]), c01));
  _mm512_store_ps(&C[ldc*2], _mm512_add_ps(_mm512_load_ps(&C[ldc*2]), c02));
  _mm512_store_ps(&C[ldc*3], _mm512_add_ps(_mm512_load_ps(&C[ldc*3]), c03));
  _mm512_store_ps(&C[ldc*0 + 16], _mm512_add_ps(_mm512_load_ps(&C[ldc*0 + 16]), c10));
  _mm512_store_ps(&C[ldc*1 + 16], _mm512_add_ps(_mm512_load_ps(&C[ldc*1 + 16]), c11));
  _mm512_store_ps(&C[ldc*2 + 16], _mm512_add_ps(_mm512_load_ps(&C[ldc*2 + 16]), c12));
  _mm512_store_ps(&C[ldc*3 + 16], _mm512_add_ps(_mm512_load_ps(&C[ldc*3 + 16]), c13));
  _mm512_store_ps(&C[ldc*0 + 32], _mm512_add_ps(_mm512_load_ps(&C[ldc*0 + 32]), c20));
  _mm512_store_ps(&C[ldc*1 + 32], _mm512_add_ps(_mm512_load_ps(&C[ldc*1 + 32]), c21));
  _mm512_store_ps(&C[ldc*2 + 32], _mm512_add_ps(_mm512_load_ps(&C[ldc*2 + 32]), c22));
  _mm512_store_ps(&C[ldc*3 + 32], _mm512_add_ps(_mm512_load_ps(&C[ldc*3 + 32]), c23));
  _mm512_store_ps(&C[ldc*0 + 48], _mm512_add_ps(_mm512_load_ps(&C[ldc*0 + 48]), c30));
  _mm512_store_ps(&C[ldc*1 + 48], _mm512_add_ps(_mm512_load_ps(&C[ldc*1 + 48]), c31));
  _mm512_store_ps(&C[ldc*2 + 48], _mm512_add_ps(_mm512_load_ps(&C[ldc*2 + 48]), c32));
  _mm512_store_ps(&C[ldc*3 + 48], _mm512_add_ps(_mm512_load_ps(&C[ldc*3 + 48]), c33));
}
static void do_block_optfill(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  float* BB = (float*)_mm_malloc(K * 8 * sizeof(float), 64);
  float* CC = (float*)_mm_malloc(32 * 8 * sizeof(float), 64);
  Pack_Array(B, BB, ldb, K, K, 8);
  Pack_Array(C, CC, ldc, 32, M, N);
  do_block_avx32x8(32, K, 32, K, A, BB, CC);
  Pack_Array(CC, C, 32, ldc, M, N);
  _mm_free(BB);
  _mm_free(CC);
}
static void do_block_optcol(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  float* BB = (float*)_mm_malloc(K * 8 *sizeof(float), 64);
  float* CC = (float*)_mm_malloc(32 * 8 *sizeof(float), 64);
  Pack_Array(B, BB, ldb, K, K, 8);
  for(int i = 0; i < M; i +=32)
  {
    for(int jj = 0; jj < N; jj ++)
    {
      _mm512_store_ps(&CC[jj*32], _mm512_load_ps(&C[i+jj*ldc]));
      _mm512_store_ps(&CC[jj*32+16], _mm512_load_ps(&C[i+16+jj*ldc]));
    }
    do_block_avx32x8(lda, K, 32, K, A+i, BB, CC);
    for(int jj = 0; jj < N; jj ++)
    {
      _mm512_store_ps(&C[i+jj*ldc], _mm512_load_ps(&CC[jj*32]));
      _mm512_store_ps(&C[i+16+jj*ldc], _mm512_load_ps(&CC[16+jj*32]));
    }
  }
  _mm_free(BB);
  _mm_free(CC);
}
static void do_block_optsta(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C) 
{
  float* CC = (float*)_mm_malloc(32 * 8 * sizeof(float), 64);
  for(int j = 0; j < N; j += 8)
  {
    Pack_Array(C+j*ldc, CC, ldc, 32, M, 8);
    do_block_avx32x8(32, ldb, 32, K, A, B+j*ldb, CC);
    Pack_Array(CC, C+j*ldc, 32, ldc, M, 8);
  }
  _mm_free(CC);
}
static void do_block_gebp(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  // do 64 * 16 block
  if((M % 64 == 0) && (N % 16 == 0))
  {
    for(int j = 0; j < N; j += 16)
    {
      for(int i = 0; i < M; i += 64)
      {
        do_block_avx64x4(lda, ldb, ldc, K, A+i, B+(j+0)*ldb, C+i+(j+0)*ldc);
        do_block_avx64x4(lda, ldb, ldc, K, A+i, B+(j+4)*ldb, C+i+(j+4)*ldc);
        do_block_avx64x4(lda, ldb, ldc, K, A+i, B+(j+8)*ldb, C+i+(j+8)*ldc);
        do_block_avx64x4(lda, ldb, ldc, K, A+i, B+(j+12)*ldb, C+i+(j+12)*ldc);
      }
    }
  }
  // else do 32 * 8 block
  else if((M % 32 == 0) && (N % 8 == 0))
  {
    for (int j = 0; j < N; j += 8)
      for (int i = 0; i < M; i += 32)
        do_block_avx32x8(lda, ldb, ldc, K, A+i, B+j*ldb, C+i+j*ldc);
    return;
  }
  // else blocking into 4 squares with m*32 x n*8, m*32 x remainder_n( < 8), remainder_m( < 32) x n*8, remainder_m x remainder_n.
  else
  { 
    int remainder_M = M % 32;
    int remainder_N = N % 8;
    int peel_M = M - remainder_M;
    int peel_N = N - remainder_N;
    if(peel_M > 0)
    {
      if(peel_N > 0)
        do_block_gebp(lda, ldb, ldc, peel_M, peel_N, K, A, B, C);
      if(remainder_N > 0)
        // using the do_block_opt_column kernel.
        do_block_optcol(lda, ldb, ldc, peel_M, remainder_N, K, A, B+peel_N*ldb, C+peel_N*ldc);
    }
    if(remainder_M > 0)
    {
      float *AA = (float*)_mm_malloc(32 * K * sizeof(float), 64);
      for(int kk = 0; kk < K; kk++)
      {
        _mm512_store_ps(&AA[kk*32], _mm512_load_ps(&A[kk*lda + peel_M]));
        _mm512_store_ps(&AA[kk*32+16], _mm512_load_ps(&A[kk*lda + peel_M +16]));
      }
      if(peel_N > 0)
      // using the do_block_opt_stack kernel
        do_block_optsta(32, ldb, ldc, remainder_M, peel_N, K, AA, B, C+peel_M);
      if(remainder_N > 0)
      // using the do_block_opt_fillremainder kernel.
        do_block_optfill(32, ldb, ldc, remainder_M, remainder_N, K, AA, B+peel_N*ldb, C+peel_N*ldc+peel_M);
      _mm_free(AA);
    }
  }
}
static void do_block_smaller (int lda, int M, int N, int K, float *A, float *B, float *C)
{
  if((M % 32 == 0) && (N % 8 == 0))
  {
    for (int j = 0; j < N; j += 8)
      for (int i = 0; i < M; i += 32)
        do_block_avx32x8 (lda, lda, lda, K, A+i, B+j*lda, C+i+j*lda);
    return;
  }
  else
  { 
    int remainder_M = M % 32;
    int remainder_N = N % 8;
    int peel_M = M - remainder_M;
    int peel_N = N - remainder_N;
    if(peel_M > 0)
    {
      if(peel_N > 0)
      {
        for(int j = 0; j < peel_N; j += 8)
          for(int i = 0; i < peel_M; i += 32)
          do_block_avx32x8 (lda, lda, lda, K, A+i, B+j*lda, C+i+j*lda); 
      }
      if(remainder_N > 0)
      {
        do_block_optcol(lda, lda, lda, peel_M, remainder_N, K, A, B+peel_N*lda, C+peel_N*lda);
      }
    }
    if(remainder_M  > 0)
    { 
      float *AA = (float*)_mm_malloc(32 * K * sizeof(float), 64);
      for(int kk = 0; kk < K; kk++)
      {
        _mm512_store_ps(&AA[kk*32], _mm512_load_ps(&A[kk*lda + peel_M]));
        _mm512_store_ps(&AA[kk*32+16], _mm512_load_ps(&A[kk*lda + peel_M + 16]));
      }
      if(peel_N > 0)
      {
        do_block_optsta(32, lda, lda, remainder_M, peel_N, K, AA, B, C+peel_M);
      }
      if(remainder_N > 0)
      {
        do_block_optfill(32, lda, lda, remainder_M, remainder_N, K, AA, B+peel_N*lda, C+peel_M+peel_N*lda);
      }
      _mm_free(AA);
    }
  }
}
/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_sgemm(int lda, float* A, float* B, float* C) {
  if (lda > 100) {
      // only change the size m dynamically.
      int current_block_size_m = BLOCK_SIZE_M;
      if (lda > 250)
      {
        current_block_size_m = 384;
      }
      if (lda > 500)
      {
        current_block_size_m = 512;
      }
      float* A_packed = (float*)_mm_malloc(sizeof(float) * current_block_size_m * BLOCK_SIZE_K, 256);
      float* B_packed = (float*)_mm_malloc(sizeof(float) * BLOCK_SIZE_K * BLOCK_SIZE_N, 256);

      for (int k = 0; k < lda; k += BLOCK_SIZE_K) {
          int K = min(BLOCK_SIZE_K, lda - k);
          for (int i = 0; i < lda; i += current_block_size_m) {
              int M = min(current_block_size_m, lda - i);
              // prefetch the next A and make sure the A_packed in the L2 cache.
              if (i + current_block_size_m < lda) {
                  _mm_prefetch((char*)(A + (i + current_block_size_m) + k * lda), _MM_HINT_T1);
              }
              Pack_Array(A + i + k * lda, A_packed, lda, current_block_size_m, M, K);
              for (int j = 0; j < lda; j += BLOCK_SIZE_N) {
                  int N = min(BLOCK_SIZE_N, lda - j);
                  // prefetch the next B and make sure the B_packed in the L1 cache.
                  if (j + BLOCK_SIZE_N < lda) {
                      _mm_prefetch((char*)(B + k + (j + BLOCK_SIZE_N) * lda), _MM_HINT_T0);
                  }
                  Pack_Array(B + k + j * lda, B_packed, lda, BLOCK_SIZE_K, K, N);
                  do_block_gebp(current_block_size_m, BLOCK_SIZE_K, lda, M, N, K, A_packed, B_packed, C + i + j * lda);
              }
          }
      }
      _mm_free(A_packed);
      _mm_free(B_packed);
  } 
  else
  {
    for (int k = 0; k < lda; k += BLOCK_SIZE)
      for (int j = 0; j < lda; j += BLOCK_SIZE)
        /* Accumulate block sgemms into block of C */
        for (int i = 0; i < lda; i += BLOCK_SIZE)
        {
    /* Correct block dimensions if block "goes off edge of" the matrix */
    int M = min (BLOCK_SIZE, lda-i);
    int N = min (BLOCK_SIZE, lda-j);
    int K = min (BLOCK_SIZE, lda-k);

    /* Perform individual block sgemm */
    do_block_smaller(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
        }
  }
}
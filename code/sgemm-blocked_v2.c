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
  float *array_packed, *array_old;
  array_packed = Array_packed;
  array_old = Array_old;
  int jj,kk;
  for(jj = 0; jj < SECOND_BLOCK_SIZE; jj++)
  {
    for(kk = 0; kk < FIRST_BLOCK_SIZE - 15; kk+=16)
      _mm512_store_ps(&array_packed[kk+jj*lda_new], _mm512_load_ps(&array_old[kk+jj*lda]));
    for(; kk < FIRST_BLOCK_SIZE; kk++)
      array_packed[kk+jj*lda_new] = array_old[kk+jj*lda];
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
static void do_block_opt_old (int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
    {
      register float bkj = B[k+j*ldb];
      # pragma vector aligned
      # pragma unroll(4)
      for (int i = 0; i < M; ++i)
      C[i+j*ldc] += A[i+k*lda] * bkj;
    }
}
static void do_block_opt(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  for(int k = 0; k < K; ++k)
  {
    for(int j = 0; j < N; ++j)
    {
      register float b = B[k+j*ldb];
      int i = 0;
      # pragma ivdep
      for(; i <= M - 16; i += 16)
      {
        __m512 a = _mm512_load_ps(&A[i+k*lda]);
        __m512 c = _mm512_load_ps(&C[i+j*ldc]);
        c = _mm512_fmadd_ps(a,_mm512_set1_ps(b),c);
        _mm512_store_ps(&C[i+j*ldc], c);
      }
      if(i < M)
      {
        __mmask16 mask = (1 << (M-i)) - 1;
        __m512 a = _mm512_maskz_load_ps(mask, &A[i+k*lda]);
        __m512 c = _mm512_maskz_load_ps(mask, &C[i+j*ldc]);
        c = _mm512_fmadd_ps(a, _mm512_set1_ps(b), c);
        _mm512_mask_store_ps(&C[i+j*ldc], mask, c);
      }
    }
  }
}
static void do_block_optfill(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  float* AA = (float*)_mm_malloc(32 * K * sizeof(float), 64);
  float* BB = (float*)_mm_malloc(K * 8 * sizeof(float), 64);
  float* CC = (float*)_mm_malloc(32 * 8 * sizeof(float), 64);
  for(int kk = 0; kk < K; kk++)
  {
    _mm512_store_ps(&AA[kk*32], _mm512_load_ps(&A[kk*lda]));
    _mm512_store_ps(&AA[kk*32+16], _mm512_load_ps(&A[kk*lda+16]));
  }
  Pack_Array(B, BB, ldb, K, K, 8);
  for(int jj = 0; jj < N; jj++)
  {
    for(int ii = 0; ii < M; ii++)
      CC[ii + jj * 32] = C[ii + jj * ldc];
  }
  do_block_avx32x8(32, K, 32, K, AA, BB, CC);
  for(int jj = 0; jj < N; jj++)
    for(int ii = 0; ii < M; ii++)
      C[ii+jj*ldc] = CC[ii+jj*32];
  _mm_free(AA);
  _mm_free(BB);
  _mm_free(CC);
}
static void do_block_optcol(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  // for M is peel_M which is the m*32. N is remainder_N which is below than 8
  for(int k = 0; k < K; k++)
  {
    for(int j = 0; j < N; j++)
    {
      __m512 b = _mm512_set1_ps(B[k+j*ldb]);
      for(int i = 0; i < M; i += 32)
      {
        __m512 a0 = _mm512_load_ps(&A[i+k*lda]);
        __m512 a1 = _mm512_load_ps(&A[i+16+k*lda]);
        __m512 c0 = _mm512_load_ps(&C[i+j*ldc]);
        __m512 c1 = _mm512_load_ps(&C[i+16+j*ldc]);
        c0 = _mm512_fmadd_ps(a0, b, c0);
        c1 = _mm512_fmadd_ps(a1, b, c1);
        _mm512_store_ps(&C[i+j*ldc],c0);
        _mm512_store_ps(&C[i+16+j*ldc],c1);
      }
    }
  }
}
static void do_block_optsta_old(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C) 
{
    const int block_size = 8; // 每次处理8列
    for (int j = 0; j < N; j += block_size) {
        float* BB_block = (float*)_mm_malloc(K * block_size * sizeof(float), 32);
        for (int k = 0; k < K; k++) {
            for (int jj = 0; jj < block_size; jj++) {
                BB_block[k * block_size + jj] = B[k + (j + jj) * ldb];
            }
        }
        float* CC_block = (float*)_mm_malloc(M * block_size * sizeof(float), 32);
        for (int i = 0; i < M; i++) {
            for (int jj = 0; jj < block_size; jj++) {
                CC_block[i * block_size + jj] = C[i + (j + jj) * ldc];
            }
        }
        for (int i = 0; i < M; i++) {
            __m256 c_vec = _mm256_load_ps(&CC_block[i * block_size]);
            for (int k = 0; k < K; k++) 
            {
                __m256 a = _mm256_set1_ps(A[i+k*lda]);
                __m256 b_vec = _mm256_load_ps(&BB_block[k * block_size]);
                c_vec = _mm256_fmadd_ps(a, b_vec, c_vec);
            }
            _mm256_store_ps(&CC_block[i * block_size], c_vec);
        }
        for (int i = 0; i < M; i++) {
            for (int jj = 0; jj < block_size; jj++) {
                C[i + (j + jj) * ldc] = CC_block[i * block_size + jj];
            }
        }
        _mm_free(BB_block);
        _mm_free(CC_block);
    }
}
static void do_block_optsta(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C) 
{
  float* AA = (float*)_mm_malloc(32 * K * sizeof(float), 64);
  float* CC = (float*)_mm_malloc(32 * 8 * sizeof(float), 64);
  for(int kk = 0; kk < K; kk++)
  {
    _mm512_store_ps(&AA[kk*32], _mm512_load_ps(&A[kk*lda]));
    _mm512_store_ps(&AA[kk*32+16], _mm512_load_ps(&A[kk*lda+16]));
  }
  for(int j = 0; j < N; j += 8)
  {
    for(int jj = 0; jj < 8; jj++)
    {
      for(int ii = 0; ii < M; ii++)
        CC[ii + jj * 32] = C[ii + (jj+j) * ldc];
    }
    do_block_avx32x8(32, ldb, 32, K, AA, B+j*ldb, CC);
    for(int jj = 0; jj < 8; jj++)
    {
      for(int ii = 0; ii < M; ii++)
        C[ii + (jj + j) * ldc] = CC[ii + jj * 32];
    }
  }
  _mm_free(AA);
  _mm_free(CC);
}
static void do_block_gebp(int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
{
  // do 32 * 8 block
  if((M % 32 == 0) && (N % 8 == 0))
  {
    for (int j = 0; j < N; j += 8)
      for (int i = 0; i < M; i += 32)
        do_block_avx32x8(lda, ldb, ldc, K, A+i, B+j*ldb, C+i+j*ldc);
    return;
  }
  else
  { 
    int remainder_M = M % 32;
    int remainder_N = N % 8;
    int peel_M = M - remainder_M;
    int peel_N = N - remainder_N;
    int i,j;
    if(peel_M > 0)
    {
      if(peel_N > 0)
      {
        for(int j = 0; j < peel_N; j += 8)
        for(int i = 0; i < peel_M; i += 32)
          do_block_avx32x8(lda, ldb, ldc, K, A+i, B+j*ldb, C+i+j*ldc);
      }
      if(remainder_N > 0)
        // using the do_block_opt_col kernel.
        do_block_optcol(lda, ldb, ldc, peel_M, remainder_N, K, A, B+peel_N*ldb, C+peel_N*ldc);
    }
    if(remainder_M > 0)
    {
      if(peel_N > 0)
      // using the do_block_opt_avx2 and the transition.
        do_block_optsta(lda, ldb, ldc, remainder_M, peel_N, K, A+peel_M, B, C+peel_M);
      if(remainder_N > 0)
        do_block_optfill(lda, ldb, ldc, remainder_M, remainder_N, K, A+peel_M, B+peel_N*ldb, C+peel_N*ldc+peel_M);
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
    int i,j;
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
      if(peel_N > 0)
      {
        do_block_optsta(lda, lda, lda, remainder_M, peel_N, K, A+peel_M, B, C+peel_M);
      }
      if(remainder_N > 0)
      {
        do_block_optfill(lda, lda, lda, remainder_M, remainder_N, K, A+peel_M, B+peel_N*lda,C+peel_M+peel_N*lda);
      }
    }
  }
}
/* This routine performs a sgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_sgemm (int lda, float* A, float* B, float* C)
{
  if (lda > 100)
  {
    float* A_packed = (float*)_mm_malloc(sizeof(float)* BLOCK_SIZE_M * BLOCK_SIZE_K, 256);
    float* B_packed = (float*)_mm_malloc(sizeof(float)* BLOCK_SIZE_K * BLOCK_SIZE_N, 256);
    for (int k = 0; k < lda; k += BLOCK_SIZE_K)
    {
      int K = min (BLOCK_SIZE_K, lda-k);
      for (int i = 0; i < lda; i += BLOCK_SIZE_M)
      {
        int M = min (BLOCK_SIZE_M, lda-i);
        if (i+BLOCK_SIZE_M < lda)
        {
          _mm_prefetch((char*)(A+(i+BLOCK_SIZE_M)+k*lda),_MM_HINT_T1);
        }
        // packing A_block into L2 cache;
        Pack_Array(A+i+k*lda, A_packed, lda, BLOCK_SIZE_M, M, K);
        for (int j = 0; j < lda; j += BLOCK_SIZE_N)
        {
          int N = min (BLOCK_SIZE_N, lda-j);
          if (j+BLOCK_SIZE_N < lda)
          {
            _mm_prefetch((char*)(B+k+(j+BLOCK_SIZE_N)*lda),_MM_HINT_T0);
          }
          //  packing B_row_block, and prepare for the B_line_block packed into L1 cache
          Pack_Array(B+k+j*lda, B_packed, lda, BLOCK_SIZE_K, K, N);
          // 考虑是否只在这里引入对B的大分割，而对B的小分割和C的小分割在do_block函数中完成。
          do_block_gebp(BLOCK_SIZE_M, BLOCK_SIZE_K, lda, M, N, K, A_packed, B_packed, C+i+j*lda);
          // 在do_block中对块进一步切割，令B被切割后的小块放在L1cache中。
        }
      }
    }
    _mm_free(A_packed);
    _mm_free(B_packed);
  }
  else
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

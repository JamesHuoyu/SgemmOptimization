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
      _mm512_storeu_ps(&array_packed[kk+jj*lda_new], _mm512_loadu_ps(&array_old[kk+jj*lda]));
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
    __m512 a0 = _mm512_loadu_ps(&A[k*lda]);
    __m512 a1 = _mm512_loadu_ps(&A[k*lda+16]);
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
  _mm512_storeu_ps(&C[ldc*0], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*0]), c00));
  _mm512_storeu_ps(&C[ldc*0+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*0+16]), c01));
  _mm512_storeu_ps(&C[ldc*1], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*1]), c10));
  _mm512_storeu_ps(&C[ldc*1+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*1+16]), c11));
  _mm512_storeu_ps(&C[ldc*2], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*2]), c20));
  _mm512_storeu_ps(&C[ldc*2+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*2+16]), c21));
  _mm512_storeu_ps(&C[ldc*3], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*3]), c30));
  _mm512_storeu_ps(&C[ldc*3+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*3+16]), c31));
  _mm512_storeu_ps(&C[ldc*4], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*4]), c40));
  _mm512_storeu_ps(&C[ldc*4+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*4+16]), c41));
  _mm512_storeu_ps(&C[ldc*5], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*5]), c50));
  _mm512_storeu_ps(&C[ldc*5+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*5+16]), c51));
  _mm512_storeu_ps(&C[ldc*6], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*6]), c60));
  _mm512_storeu_ps(&C[ldc*6+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*6+16]), c61));
  _mm512_storeu_ps(&C[ldc*7], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*7]), c70));
  _mm512_storeu_ps(&C[ldc*7+16], _mm512_add_ps(_mm512_loadu_ps(&C[ldc*7+16]), c71));
}
static void do_block_opt (int lda, int ldb, int ldc, int M, int N, int K, float* __restrict A, float* __restrict B, float* __restrict C)
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
    int flag = 0;
    int remainder_M = M;
    int remainder_N = N;
    int peel_M = 0;
    int peel_N = 0;
    if ((remainder_M > 32)&&(remainder_N > 8))
    {
      flag = 1;
      remainder_M = M % 32;
      remainder_N = N % 8;
      peel_M = M - remainder_M;
      peel_N = N - remainder_N;
    }
    int i,j;
    if(flag == 1)
    {
      for(int j = 0; j < peel_N; j += 8)
        for(int i = 0; i < peel_M; i += 32)
          do_block_avx32x8(lda, ldb, ldc, K, A+i, B+j*ldb, C+i+j*ldc);
    }
    if(remainder_M > 0)
      do_block_opt(lda, ldb, ldc, remainder_M, N, K, A+peel_M, B, C+peel_M);
    if((remainder_N > 0)&&(peel_M > 0))
      do_block_opt(lda, ldb, ldc, peel_M, remainder_N, K, A, B+peel_N*ldb, C+peel_N*ldc);
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
    int flag = 0;
    int remainder_M = M;
    int remainder_N = N;
    int peel_M = 0;
    int peel_N = 0;
    if ((remainder_M > 32)&&(remainder_N > 8))
    {
      flag = 1;
      remainder_M = M % 32;
      remainder_N = N % 8;
      peel_M = M - remainder_M;
      peel_N = N - remainder_N;
    }
    int i,j;
    if(flag == 1)
    {
      for(int j = 0; j < peel_N; j += 8)
        for(int i = 0; i < peel_M; i += 32)
        do_block_avx32x8 (lda, lda, lda, K, A+i, B+j*lda, C+i+j*lda);
    }
    if(remainder_M > 0)
      do_block_opt(lda, lda, lda, remainder_M, N, K, A+peel_M, B, C+peel_M);
    if((remainder_N > 0)&&(peel_M > 0))
      do_block_opt(lda, lda, lda, peel_M, remainder_N, K, A, B+peel_N*lda, C+peel_N*lda);
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
        // packing A_block into L2 cache;
        Pack_Array(A+i+k*lda, A_packed, lda, M, M, K);
        for (int j = 0; j < lda; j += BLOCK_SIZE_N)
        {
          int N = min (BLOCK_SIZE_N, lda-j);
          //  packing B_row_block, and prepare for the B_line_block packed into L1 cache
          Pack_Array(B+k+j*lda, B_packed, lda, K, K, N);
          // 考虑是否只在这里引入对B的大分割，而对B的小分割和C的小分割在do_block函数中完成。
          do_block_gebp(M, K, lda, M, N, K, A_packed, B_packed, C+i+j*lda);
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

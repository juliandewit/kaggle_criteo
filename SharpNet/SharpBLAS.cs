using Cudafy.Host;
using Cudafy.Maths.BLAS;
using Cudafy.Maths.BLAS.Types;
using Cudafy.Maths.SPARSE;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class SharpBLAS
    {
        public GPGPU Gpu;
        public GPGPUBLAS Blas;
        public GPGPUSPARSE Sparse;

        public SharpBLAS(GPGPU gpu)
        {
            Gpu = gpu;
            Blas = GPGPUBLAS.Create(gpu);
            Sparse = GPGPUSPARSE.Create(gpu);
        }

        //http://peterwittek.com/2013/06/cublas-matrix-c-style/
        //row major to col major "trick"
        public void GemmRowMajor(CpuGpuArray A, CpuGpuArray B, CpuGpuArray C,  float cMultiplier = 0f, bool transposeA = false, bool transposeB = false)
        {
            var blasA = B;
            var blasB = A;
            // M = rowcount A(T),C
            // N = colcount B(T),C
            // K = colcount A(T), rowcount B(T)
            // However, we flip so 
            // M = colcount B(T),C
            // N = rowcoun A(T), C
            // K = cols A(T), rows B
            var m = blasA.ColCount;
            var n = blasB.RowCount;
            var k = blasB.ColCount;

            var lda = blasA.ColCount;
            var ldb = blasB.ColCount;
            var ldc = blasA.ColCount;

            var transb = cublasOperation.N;
            if (transposeA)
            {
                transb = cublasOperation.T;
                n = blasB.ColCount;
                k = blasB.RowCount;
            }

            var transa = cublasOperation.N;
            if (transposeB)
            {
                transa = cublasOperation.T;
                m = blasA.RowCount;
                ldc = blasA.RowCount;
            }

            Blas.GEMM(m, k, n, 1f, B.GPUArray, A.GPUArray, cMultiplier, C.GPUArray, lda: lda, ldb: ldb, ldc: ldc, transb:transb, transa: transa);
        }
    }
}

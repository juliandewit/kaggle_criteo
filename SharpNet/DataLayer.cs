using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class DataLayer : Layer
    {
        public int BatchesPerLoad = 1;
        public int CurrentBatchNo = 0;
        public bool IsSparse = false;
        public int SparseDataSize = 0;
        public CpuGpuMatrixSparse SparseMatrix;

        public DataLayer(GPUModule gpuModule, int size, int batchesPerLoad = 1, int miniBatchSize = 128, int sparseDataSize = 0) : base(gpuModule, size: 0, miniBatchSize: miniBatchSize)
        {
            this.Size = size;
            this.IsSparse = sparseDataSize!=0;
            this.SparseDataSize = sparseDataSize;
            if (!IsSparse)
            {
                AddArray(ArrayName.Outputs, batchesPerLoad * MinibatchSize, size);
            }
            else
            {
            }
            AddArray(ArrayName.Noise, batchesPerLoad * MinibatchSize, size);
        }

        public override void CopyToGpu()
        {
            base.CopyToGpu();
        }

        public void SetData(float[] data, float noiseMean = 0f, float noiseStdDev = 0f)
        {
            if (this.OutputsCPU.Length != data.Length) throw new ArgumentException();
            Outputs.CPUArray = data;
            Outputs.CopyToGpu();

            if (noiseStdDev != 0f)
            {
                var arr = GetArray(ArrayName.Noise);
                _gpuModule.Rand.GenerateNormal(arr.GPUArray, noiseMean, noiseStdDev, n: arr.Length);
                _gpuModule.Blas.Blas.AXPY(1f, arr.GPUArray, Outputs.GPUArray);
            }
        }

        public void SetSparseData(List<float[]> values, List<int[]> indices)
        {
            if ((SparseMatrix == null))
            {
                Console.WriteLine("Creating sparse input arrays");
                SparseMatrix = new CpuGpuMatrixSparse(_gpuModule, this.SparseDataSize, MinibatchSize, this.Size);
            }

            SparseMatrix.SetData(values, indices);
            SparseMatrix.CopyToGPU();
        }

        float[] _orgGpuData = null;
        public void SetGPUData(float[] data)
        {
            if (_orgGpuData == null) _orgGpuData = this.Outputs.GPUArray;
            this.Outputs.GPUArray = data;
        }

        public void IncBatchNo()
        {
            CurrentBatchNo++;
            if (CurrentBatchNo >= BatchesPerLoad) CurrentBatchNo = 0;
        }

        public override void Free()
        {
            if (_orgGpuData != null) Outputs.GPUArray = _orgGpuData;
            base.Free();
        }

    }
}

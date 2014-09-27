using Cudafy.Host;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class CpuGpuMatrixSparse
    {
        GPUModule _gpuModule;
        public int[] CPUIndices;
        public float[] CPUValues;
        public int[] GPUIndices;
        public float[] GPUValues;
        public int DataSize;
        public int RowCount;
        public int ColCount;
        int _nonZeroCount;

        public GPGPU Gpu
        {
            get {
                if (_gpuModule == null) return null;
                return _gpuModule.Gpu; 
            }
        }

        public CpuGpuMatrixSparse(GPUModule module, int dataSize, int rowCount, int colCount)
        {
            _gpuModule = module;
            ColCount = colCount;
            InitArrays(dataSize, rowCount);
        }

        public CpuGpuMatrixSparse(GPUModule module, int dataSize, List<float[]> values, List<int[]> indices, int targetColCount)
        {
            _gpuModule = module;
            if (values.Count != indices.Count) throw new ArgumentException();
            RowCount = values.Count;
            ColCount = targetColCount;
            DataSize = dataSize;
            NonZeroCount = values.Sum(x=>x.Length);
            InitArrays(DataSize, RowCount);
            SetData(values, indices);
        }

        public int NonZeroCount
        {
            get
            {
                return _nonZeroCount;
            }
            set
            {
                if (value > DataSize) throw new ArgumentException();
                _nonZeroCount = value;
            }
        }

        public float this[int row, int col]
        {
            get { return GetValue(row, col); }
        }

        public void SetData(List<float[]> values, List<int[]> indices)
        {
            var nonZeroIndex = 0;
            for (var rowNo = 0; rowNo < values.Count; rowNo++)
            {
                var valueArray = values[rowNo];
                var indexArray = indices[rowNo];
                if (valueArray.Length != indexArray.Length) throw new ArgumentException();
                var startIndex = rowNo * ColCount;
                for (var i = 0; i < valueArray.Length; i++)
                {
                    var colIndex = startIndex + indexArray[i];
                    var value = valueArray[i];
                    CPUValues[nonZeroIndex] = value;
                    CPUIndices[nonZeroIndex] = colIndex;
                    nonZeroIndex++;
                }
            }
            NonZeroCount = nonZeroIndex;
        }

        float GetValue(int row, int col)
        {
            var wantedIndex = row * ColCount + col;
            for (var i = 0; i < NonZeroCount; i++)
            {
                var index = (int)CPUIndices[i];
                if (index == wantedIndex) return CPUValues[index];
                if (index > wantedIndex) return 0f;
            }
            return 0f;
        }

        void InitArrays(int dataSize, int rowCount)
        {
            if (NonZeroCount > dataSize) throw new ArgumentException();
            DataSize = dataSize;
            RowCount = rowCount;
            CPUValues = new float[DataSize];
            CPUIndices = new int[DataSize];

            if (_gpuModule != null)
            {
                GPUValues = CpuGpuArray.AllocateGPUArray(_gpuModule, DataSize);
                GPUIndices = CpuGpuArrayInt.AllocateGPUArray(_gpuModule, DataSize);
            }
        }

        public void CopyToGPU()
        {
            Gpu.CopyToDevice(CPUValues, GPUValues);
            Gpu.CopyToDevice(CPUIndices, GPUIndices);
        }

        public void Free()
        {
            Gpu.Free(GPUValues);
            Gpu.Free(GPUIndices);
        }

    }
}

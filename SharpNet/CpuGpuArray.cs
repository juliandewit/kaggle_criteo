using Cudafy;
using Cudafy.Host;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class CpuGpuArray
    {
        GPUModule _gpuModule;
        public float[] CPUArray;
        public float[] GPUArray;
        public int RowCount;
        public int ColCount;

        public GPGPU Gpu
        {
            get { return _gpuModule.Gpu; }
        }

        public CpuGpuArray(GPUModule gpuModule, int size)
        {
            _gpuModule = gpuModule;
            CPUArray = new float[size];
            GPUArray = AllocateGPUArray(_gpuModule, size);
            RowCount = 1;
            ColCount = size;
        }

        public CpuGpuArray(GPUModule gpuModule, float[] gpuArray, int rows, int cols, bool createCpuData = true)
        {
            _gpuModule = gpuModule;
            if (createCpuData)
            {
                CPUArray = new float[rows * cols];
            }
            GPUArray = gpuArray;
            RowCount = rows;
            ColCount = cols;
        }

        public CpuGpuArray(CpuGpuArray array, int rows, int cols)
        {
            _gpuModule = array._gpuModule;
            CPUArray = array.CPUArray;
            GPUArray = array.GPUArray;
            RowCount = rows;
            ColCount = cols;
        }


        public CpuGpuArray(GPUModule gpuModule, int rows, int cols)
            : this(gpuModule, rows * cols)
        {
            RowCount = rows;
            ColCount = cols;
        }

        public int Length
        {
            get
            {
                return CPUArray.Length;
            }
        }

        public int Width
        {
            get { return ColCount; }
        }

        public int Height
        {
            get { return RowCount; }
        }
        
        public float this[int index]
        {
            get { return this.CPUArray[index]; }
            set { this.CPUArray[index] = value; }
        }

        public float this[int row, int col]
        {
            get { return this[row * ColCount + col]; }
            set { this[row * ColCount + col] = value; }
        }

        public static void ClearGpuArray(GPUModule module, float[] gpuArray, int size)
        {
            //var array = new float[size];
            //Array.Clear(array, 0, array.Length);
            //gpu.CopyToDevice(array, gpuArray);
            module.FillArrayRaw(gpuArray, size, 0f);
        }

        public static void ClearGpuArray(GPGPU gpu, int[] gpuArray, int size)
        {
            var array = new int[size];
            Array.Clear(array, 0, array.Length);
            gpu.CopyToDevice(array, gpuArray);
        }

        public void FillGpu(float value)
        {
            _gpuModule.FillArrayRaw(this.GPUArray, this.Length, value);
        }

        public static int[] AllocateGPUArrayInt(GPUModule module, int size)
        {
            var res = module.Gpu.Allocate<int>(size);
            ClearGpuArray(module.Gpu, res, size);
            return res;
        }

        public static float[] AllocateGPUArray(GPUModule module, int size)
        {
            var res = module.Gpu.Allocate<float>(size);
            ClearGpuArray(module, res, size);
            return res;
        }

        public void CopyToGpu()
        {
            Gpu.CopyToDevice(CPUArray, GPUArray);
        }

        public void CopyToHost()
        {
            Gpu.CopyFromDevice(GPUArray, CPUArray);
        }

        public void Free()
        {
            Gpu.Free(GPUArray);
        }

        public void InitValuesRandomCPU(float mean, float std)
        {
            for (var i = 0; i < this.Length; i++)
            {
                var val = (float)RandomHelpers.GetRandomGaussian(mean, std);
                CPUArray[i] = val;
            }
        }

        public void InitValuesUniformCPU(float max)
        {
            for (var i = 0; i < this.Length; i++)
            {
                var val = (float)RandomHelpers.NextFloat();
                CPUArray[i] = (val * max * 2f) - max;
            }
        }

        
        public void FillCPU(float value)
        {
            for (var i = 0; i < this.Length; i++)
            {
                CPUArray[i] = value;
            }
        }

        public string GetTxt()
        {
            var builder = new StringBuilder();
            var culture = (CultureInfo)CultureInfo.InvariantCulture.Clone();
            culture.NumberFormat.NumberDecimalSeparator = ",";
            for (var row = 0; row < this.RowCount; row++)
            {
                for (var col = 0; col < this.ColCount; col++)
                {
                    var val = this[row, col];
                    builder.AppendLine(row.ToString().PadLeft(4) + "\t" + col.ToString().PadLeft(4) + "\t" + val.ToString(culture));
                }
            }
            return builder.ToString();
        }

        public void SaveBinary(string path)
        {
            var bytes = new byte[this.Length * 4];
            Buffer.BlockCopy(this.CPUArray, 0, bytes, 0, this.Length * 4);
            File.WriteAllBytes(path, bytes);
        }
        
        public void LoadBinary(string path)
        {
            var bytes = File.ReadAllBytes(path);
            if (bytes.Length != this.Length * 4) throw new Exception("Length exception !");
            Buffer.BlockCopy(bytes, 0, this.CPUArray, 0, this.Length * 4);
        }

    }
}

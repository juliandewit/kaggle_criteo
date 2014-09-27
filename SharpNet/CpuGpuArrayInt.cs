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
    public class CpuGpuArrayInt
    {
        GPUModule _gpuModule;
        public int[] CPUArray;
        public int[] GPUArray;
        public int RowCount;
        public int ColCount;

        public GPGPU Gpu
        {
            get { return _gpuModule.Gpu; }
        }

        public CpuGpuArrayInt(GPUModule gpuModule, int size)
        {
            _gpuModule = gpuModule;
            CPUArray = new int[size];
            GPUArray = AllocateGPUArray(_gpuModule, size);
            RowCount = 1;
            ColCount = size;
        }

        public CpuGpuArrayInt(GPUModule gpuModule, int[] gpuArray, int rows, int cols, bool createCpuData = true)
        {
            _gpuModule = gpuModule;
            if (createCpuData)
            {
                CPUArray = new int[rows * cols];
            }
            GPUArray = gpuArray;
            RowCount = rows;
            ColCount = cols;
        }

        public CpuGpuArrayInt(CpuGpuArrayInt array, int rows, int cols)
        {
            _gpuModule = array._gpuModule;
            CPUArray = array.CPUArray;
            GPUArray = array.GPUArray;
            RowCount = rows;
            ColCount = cols;
        }


        public CpuGpuArrayInt(GPUModule gpuModule, int rows, int cols) : this(gpuModule, rows * cols)
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
        
        public int this[int index]
        {
            get { return this.CPUArray[index]; }
            set { this.CPUArray[index] = value; }
        }

        public int this[int row, int col]
        {
            get { return this[row * ColCount + col]; }
            set { this[row * ColCount + col] = value; }
        }

        public static void ClearGpuArray(GPUModule module, int[] gpuArray, int size)
        {
            module.FillArrayRawInt(gpuArray, size, 0);
        }

        public static void ClearGpuArray(GPGPU gpu, int[] gpuArray, int size)
        {
            var array = new int[size];
            Array.Clear(array, 0, array.Length);
            gpu.CopyToDevice(array, gpuArray);
        }

        public static int[] AllocateGPUArray(GPUModule module, int size)
        {
            var res = module.Gpu.Allocate<int>(size);
            ClearGpuArray(module.Gpu, res, size);
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
                
        public void FillCPU(int value)
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

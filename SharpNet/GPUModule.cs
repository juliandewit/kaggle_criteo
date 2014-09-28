using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using Cudafy.Atomics;
using Cudafy.Maths.BLAS;
using Cudafy.Rand;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy.Maths.RAND;
using Cudafy.Maths.SPARSE;


namespace SharpNet
{
    public class GPUModule
    {
        const int MAX_THREAD_COUNT = 1024;
        const int MAX_BLOCKS_DIM = 65535;
        eGPUType _gpuType = eGPUType.Cuda;
        public GPGPU Gpu;
        public SharpBLAS Blas;
        public GPGPURAND Rand;

        public GPUModule(eGPUType gpuType = eGPUType.Cuda)
        {
            _gpuType = gpuType;
        }

        public GPGPUSPARSE Sparse
        {
            get
            {
                return Blas.Sparse;
            }
        }

        public Tuple<int, int> ComputeBlocksTreads(int count)
        {
            if (_gpuType == eGPUType.Emulator)
            {
                return Tuple.Create(1, 1);
            }

            var threads = MAX_THREAD_COUNT;
            var blocks = (count / threads) + 1;
            if (blocks > MAX_BLOCKS_DIM)
            {
                blocks = MAX_BLOCKS_DIM;
            }

            if (blocks > 65500)
            {
                Console.WriteLine("warning : many blocks ! (" + blocks + ")");
            }

            var res = Tuple.Create(blocks, threads);
            return res;
        }

        public void InitGPU()
        {
            // Work around for bug in Cudafy trying to find the path..
            var os64Bit = Environment.Is64BitOperatingSystem;
            if (os64Bit)
            {
                var dir = Environment.GetEnvironmentVariable("ProgramFiles");
                Environment.SetEnvironmentVariable("ProgramFiles", "C:\\Program Files\\");
                dir = Environment.GetEnvironmentVariable("ProgramFiles");
            }

            if (Gpu == null)
            {
                Gpu = CudafyHost.GetDevice(_gpuType, 0);
                //Blas = GPGPUBLAS.Create(Gpu);
                if (_gpuType == eGPUType.Cuda)
                {
                    Blas = new SharpBLAS(Gpu);
                    Rand = GPGPURAND.Create(Gpu, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
                    Rand.SetPseudoRandomGeneratorSeed((ulong)RandomHelpers.Next(9999));
                }

                CudafyTranslator.GenerateDebug = true;
                Debug.WriteLine("CUDA workdir = " + CudafyTranslator.WorkingDirectory);
                Console.WriteLine("Recompile module");
                CudafyTranslator.Language = eLanguage.Cuda;
                var km = CudafyTranslator.Cudafy(eArchitecture.sm_30);
                km = CudafyTranslator.Cudafy();
                km.Serialize(); 
                Gpu.LoadModule(km);
            }
        }

        public void Free(object devArray)  
        {
            if (_gpuType == eGPUType.Cuda)
            {
                Gpu.Free(devArray);
            }
        }

        public void CalculateRelu(ReluLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Outputs.Length);
            Gpu.Launch(dims.Item1, dims.Item2).CalculateReluGPU(
               layer.Inputs.GPUArray,
               layer.Outputs.GPUArray,
               layer.Outputs.Length
            );
        }

        [Cudafy]
        public static void CalculateReluGPU(GThread thread, float[] inputs, float[] outputs, int size)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < size)
            {
                outputs[index] = GMath.Max(inputs[index], 0f);
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void BackPropagateRelu(ReluLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Inputs.Length);
            Gpu.Launch(dims.Item1, dims.Item2).BackPropagateReluGPU(
               layer.Inputs.Length,
               layer.InputGradients.GPUArray,
               layer.Gradients.GPUArray,
               layer.Outputs.GPUArray
            );
        }

        [Cudafy]
        public static void BackPropagateReluGPU(GThread thread,int kernelCount, float[] inputGradients, float[] gradients,float[] outputs)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var gradient = gradients[index];
                var output = outputs[index];
                if (output <= 0) gradient = 0;
                inputGradients[index] = gradient;
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void CalculateTanh(TanhLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Outputs.Length);
            Gpu.Launch(dims.Item1, dims.Item2).CalculateTanhGPU(
               layer.Inputs.GPUArray,
               layer.Outputs.GPUArray,
               layer.Outputs.Length
            );
        }

        [Cudafy]
        public static void CalculateTanhGPU(GThread thread, float[] inputs, float[] outputs, int size)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < size)
            {
                outputs[index] = GMath.Tanh(inputs[index]);
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void BackPropagateTanh(TanhLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Inputs.Length);
            Gpu.Launch(dims.Item1, dims.Item2).BackPropagateTanhGPU(
               layer.Inputs.Length,
               layer.InputGradients.GPUArray,
               layer.Gradients.GPUArray,
               layer.Outputs.GPUArray
            );
        }


        [Cudafy]
        public static void BackPropagateTanhGPU(GThread thread, int kernelCount, float[] inputGradients, float[] gradients, float[] outputs)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var gradient = gradients[index];
                var output = outputs[index];
                //if (output <= 0) gradient = 0;
                inputGradients[index] = gradient * (1f - output * output);
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void CalculateMaxout(MaxoutLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Outputs.Length);
            Gpu.Launch(dims.Item1, dims.Item2).CalculateMaxoutGPU(
               layer.Inputs.GPUArray,
               layer.Outputs.GPUArray,
               layer.Winners.GPUArray,
               layer.Outputs.Length,
               layer.GroupSize
            );
            layer.Winners.CopyToHost();
        }

        [Cudafy]
        public static void CalculateMaxoutGPU(GThread thread, float[] inputs, float[] outputs, int[] winners, int size, int groupSize)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < size)
            {
                var startInputIdx = index * groupSize;
                var endInputIdx = startInputIdx + groupSize;
                var max = -9999f;
                var winner = -1;
                for (var inputIndex = startInputIdx; inputIndex < endInputIdx; inputIndex++)
                {
                    var value = inputs[inputIndex];
                    if (value > max)
                    {
                        max = value;
                        winner = inputIndex;
                    }
                }
                outputs[index] = max;
                winners[index] = winner;
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void BackPropagateMaxout(MaxoutLayer layer)
        {
            var dims = ComputeBlocksTreads(layer.Outputs.Length);
            layer.PreviousLayer.Gradients.FillGpu(0f);
            Gpu.Launch(dims.Item1, dims.Item2).BackPropagateMaxoutGPU(
               layer.Outputs.Length,
               layer.InputGradients.GPUArray,
               layer.Gradients.GPUArray,
               layer.Winners.GPUArray
            );
        }

        [Cudafy]
        public static void BackPropagateMaxoutGPU(GThread thread, int kernelCount, float[] inputGradients, float[] gradients, int[] winners)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var gradient = gradients[index];
                var winnerIdx = winners[index];
                inputGradients[winnerIdx] = gradient;
                index += MAX_BLOCKS_DIM * MAX_THREAD_COUNT;
            }
        }

        public void CalculateSoftmax(SoftMaxCostLayer layer)
        {
            Gpu.Launch(layer.MinibatchSize, 1).CalculateSoftmaxGPU(
                layer.InputsGPU,
                layer.Outputs.GPUArray,
                layer.Labels.GPUArray,
                layer.CorrectlyPredictedLabels.GPUArray,
                layer.Gradients.GPUArray,
                layer.Size
                );
        }

        [Cudafy]
        public static void CalculateSoftmaxGPU(GThread thread, float[] inputs, float[] outputs, float[] labels, float[] correctLabels, float[] gradients, int size)
        {
            var minibatchIdx = thread.blockIdx.x;
            var idx = thread.threadIdx.x;
            if (idx > 0) return;
            var max = float.MinValue;
            var maxIdx = -1;
            var baseIndex = minibatchIdx * size;

            for (int outputIndex = 0; outputIndex < size; outputIndex++)
            {
                var val = inputs[baseIndex + outputIndex];
                if (val > max)
                {
                    maxIdx = outputIndex;
                    max = val;
                }
            }

            var total = 0f;
            for (int outputIndex = 0; outputIndex < size; outputIndex++)
            {
                var val = inputs[baseIndex + outputIndex];
                var exp = GMath.Exp(val - max);
                total += exp;
                outputs[baseIndex + outputIndex] = exp;
                //outputs[baseIndex + outputIndex] = max;
            }

            // Already calculate cost and gradients
            var label = labels[minibatchIdx];
            correctLabels[minibatchIdx] = 0f;
            if (label == maxIdx)
            {
                correctLabels[minibatchIdx] = 1f;
            }

            for (int outputIndex = 0; outputIndex < size; outputIndex++)
            {
                var val1 = outputs[baseIndex + outputIndex];
                var val = val1 / total;
                if ((val < -1000) || (val > 1000))
                {
                    //Debug.WriteLine("Max %d, minibatch = %d, outputidx = %d", max, minibatchIdx, outputIndex);
                    Debug.WriteLine("NaN val %d", val);
                    Debug.WriteLine("NaN total %d", total);
                }

                outputs[baseIndex + outputIndex] = val;
                val = val * -1f;
                if (label == outputIndex)
                {
                    val += 1;
                }

                //if (label == 1) val *= 3f;
                gradients[baseIndex + outputIndex] = val;
            }
        }

        public void CalculateDropout(DropoutLayer layer, bool train)
        {
            var output = layer.Outputs;
            var input = layer.Inputs;
            var kernelCount = output.Length;
            var dropoutMask = layer.GetArray(ArrayName.DropoutMask);
            if (train)
            {
                this.Rand.GenerateUniform(dropoutMask.GPUArray);
            }
            var dim = ComputeBlocksTreads(kernelCount);
            var trainVal = train ? 1f : 0f;
            Gpu.Launch(dim.Item1, dim.Item2).CalculateDropoutGPU(
                kernelCount,
                input.GPUArray,
                output.GPUArray,
                dropoutMask.GPUArray,
                0.5f,
                trainVal

            );
        }

        [Cudafy]
        public static void CalculateDropoutGPU(GThread thread, int kernelCount, float[] input, float[] output, float[] dropoutMasks, float dropoutThreshold, float trainPass)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var inputValue = input[index];
                var outputValue = 0f;
                if (trainPass > 0)
                {
                    if (dropoutMasks[index] > dropoutThreshold)
                    {
                        outputValue = inputValue;
                    }
                }
                else
                {
                    outputValue = inputValue * (1f - dropoutThreshold);
                }
                output[index] = outputValue;
                index += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public void BackPropagateDropout(DropoutLayer layer)
        {
            var gradients = layer.Gradients;
            var inputGradients = layer.InputGradients;
            var kernelCount = gradients.Length;
            var dropoutMask = layer.GetArray(ArrayName.DropoutMask);
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).BPDropoutGPU(
                kernelCount,
                inputGradients.GPUArray,
                gradients.GPUArray,
                dropoutMask.GPUArray
            );
        }

        [Cudafy]
        public static void BPDropoutGPU(GThread thread, int kernelCount, float[] inputGradients, float[] gradients, float[] dropoutMasks)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var gradient = 0f;
                if (dropoutMasks[index] > 0.5f)
                {
                    gradient = gradients[index];
                }
                inputGradients[index] = gradient;
                index += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public void ElementwiseMultiplication(CpuGpuArray vector, CpuGpuArray matrix)
        {
            if (vector.Length != matrix.Height) throw new Exception("Matrix.height <> vector.length");

            var kernelCount = matrix.Length;
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).ElementwiseMultiplicationGPU(
               kernelCount,
               vector.GPUArray,
               matrix.GPUArray,
               matrix.Width
           );
        }

        [Cudafy]
        public static void ElementwiseMultiplicationGPU(GThread thread, int kernelCount, float[] vector, float[] matrix, int matrixWidth)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var row = index / matrixWidth;
                var col = index % matrixWidth;
                var matVal = matrix[(row * matrixWidth) + col];
                var vecVal = vector[row];
                var newVal = vecVal * matVal;
                matrix[(row * matrixWidth) + col] = newVal;
                index += thread.gridDim.x * thread.blockDim.x;
            }
        }

        public void SetSparseGpuValues(CpuGpuArray target, CpuGpuArray indices, CpuGpuArray values)
        {
            SetSparseGpuValues(target.GPUArray, target.RowCount, target.ColCount, indices, values);
        }
        
        //float[] indices, float[] values, int rowCount, int colCount
        public void SetSparseGpuValues(float[] gpuTarget, int targetRowCount, int targetColCount, CpuGpuArray indices, CpuGpuArray values)
        {
            if (indices.Length != values.Length) throw new ArgumentException();
            var kernelCount = indices.RowCount * indices.ColCount;
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).SetSparseGpuValuesGPU(
                kernelCount,
                gpuTarget,
                targetRowCount,
                targetColCount,
                indices.GPUArray,
                values.GPUArray,
                indices.RowCount,
                indices.ColCount
            );
        }

        [Cudafy]
        public static void SetSparseGpuValuesGPU(GThread thread, int kernelCount, float[] target, int targetRowCount, int targetColCount, float[] indices, float[] values, int rowCount, int colCount)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            if (index < kernelCount)
            {
                var row = index / colCount;
                var col = index % colCount;
                var targetStartIdx = targetColCount * row;
                var srcIdx = colCount * row + col;
                var targetColIdx = (int)indices[srcIdx];
                if (targetColIdx != -1)
                {
                    var value = values[srcIdx];
                    var targetIdx = targetStartIdx + targetColIdx;
                    target[targetIdx] = value;
                }
            }
        }

        public void MultiplySparse(CpuGpuMatrixSparse A, CpuGpuArray B, CpuGpuArray C, bool transposeA)
        {
            FillArray(C, 0f);

            var kernelCount = B.ColCount * A.NonZeroCount;
            var dim = ComputeBlocksTreads(kernelCount);
            var colcountA = A.ColCount;
            var transposeInt = transposeA ? 1 : 0;
            Gpu.Launch(dim.Item1, dim.Item2).MultiplySparseGPU2(
                kernelCount,
                A.GPUIndices,
                A.GPUValues,
                A.NonZeroCount,
                colcountA,
                B.GPUArray,
                B.ColCount,
                C.GPUArray,
                transposeInt
            );
            return;
        }

        [Cudafy]
        public static void MultiplySparseGPU2(GThread thread, int kernelCount, int[] indicesA, float[] valuesA, int nonzeroCountA, int colCountA, float[] B, int colCountB, float[] C, int transposeA)
        {
            var index = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (index < kernelCount)
            {
                var colB = index % colCountB;
                var arrayIndex = index / colCountB;
                var value = valuesA[arrayIndex];
                var indexA = indicesA[arrayIndex];
                var rowA = indexA / colCountA;
                var colA = indexA % colCountA;
                if (transposeA != 0)
                {
                    var tmp = rowA;
                    rowA = colA;
                    colA = tmp;
                }

                var valB = B[colA * colCountB + colB];
                var mul = value * valB;
                var indexC = rowA * colCountB + colB;

                thread.atomicAdd(ref C[indexC], mul);
                index += thread.blockDim.x * thread.gridDim.x;
            }
        }

        public void FillArray(CpuGpuArray array, float value)
        {
            var kernelCount = array.Length;
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).FillArrayGPU(
                kernelCount,
                array.GPUArray,
                value
            );
        }

        public void FillArrayRaw(float[] array, int length, float value)
        {
            var kernelCount = length;
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).FillArrayGPU(
                kernelCount,
                array,
                value
            );
        }

        [Cudafy]
        public static void FillArrayGPU(GThread thread, int kernelCount, float[] array, float value)
        {
            var cIndex = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (cIndex < kernelCount)
            {
                array[cIndex] = value;
                cIndex += thread.blockDim.x * thread.gridDim.x;
            }
        }

        public void FillArrayRawInt(int[] array, int length, int value)
        {
            var kernelCount = length;
            var dim = ComputeBlocksTreads(kernelCount);
            Gpu.Launch(dim.Item1, dim.Item2).FillArrayIntGPU(
                kernelCount,
                array,
                value
            );
        }

        [Cudafy]
        public static void FillArrayIntGPU(GThread thread, int kernelCount, int[] array, int value)
        {
            var cIndex = (thread.blockIdx.x * thread.blockDim.x) + thread.threadIdx.x;
            while (cIndex < kernelCount)
            {
                array[cIndex] = value;
                cIndex += thread.blockDim.x * thread.gridDim.x;
            }
        }
    }
}

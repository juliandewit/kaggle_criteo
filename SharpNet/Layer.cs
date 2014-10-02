using Cudafy.Host;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public enum ArrayName { WeightUpdates, Weights, BiasWeights, Gradients, Cost, CorrectlyPredictedLabels, Outputs, BiasMultiplier, BiasWeightUpdates, SoftmaxMax, ConvBuffer, DropoutMask, LastWeightUpdates, LastBiasWeightUpdates, Noise, Winners };
    public class Layer
    {
        public int IdCounter = 1;
        Dictionary<ArrayName, CpuGpuArray> _Arrays = new Dictionary<ArrayName, CpuGpuArray>();
        Dictionary<ArrayName, CpuGpuArrayInt> _IntArrays = new Dictionary<ArrayName, CpuGpuArrayInt>();
        public string Id = "";
        public int LayerIndex = 0;
        public int Size = 0;
        public Layer PreviousLayer;
        protected GPUModule _gpuModule;
        protected GPGPU _gpu;
        public int MinibatchSize = 128;
        public float BiasLearnRate = float.MinValue;
        public float L2Regularization = 0f;
        public int RegularizationRatio = 1;
        public int WeightUpdateCount = 0;

        public Layer(GPUModule gpuModule, Layer previousLayer = null,int size = 0, string id = "", int miniBatchSize = Int32.MinValue)
        {
            if (previousLayer != null) MinibatchSize = previousLayer.MinibatchSize;
            if (miniBatchSize != Int32.MinValue) MinibatchSize = miniBatchSize;

            LayerIndex = IdCounter++;
            Id = id;
            if (String.IsNullOrEmpty(Id))
            {
                Id = "ID" + LayerIndex.ToString().PadLeft(2, '0');
            }

            _gpuModule = gpuModule;
            _gpu = _gpuModule.Gpu;
            PreviousLayer = previousLayer;
            if (size != 0)
            {
                this.Size = size;
                AddArray(ArrayName.Outputs, MinibatchSize, this.Size);                
            }

            if ((previousLayer != null) && (size > 0))
            {
                AddArray(ArrayName.Gradients, MinibatchSize, size);
            }
        }

        public bool HasWeights
        {
            get
            {
                var weights = GetArray(ArrayName.Weights);
                return weights != null;
            }
        }

        public virtual int InputSize
        {
            get
            {
                if (PreviousLayer == null) return -1;
                return PreviousLayer.Size;
            }
        }

        public CpuGpuArray Inputs 
        {
            get
            {
                if (PreviousLayer == null) throw new ArgumentException("Previous layer is null");
                return PreviousLayer.Outputs;
            }
        }

        public float[] InputsGPU
        {
            get
            {
                return Inputs.GPUArray;
            }
        }

        public CpuGpuArray InputGradients
        {
            get
            {
                if (PreviousLayer == null) return null;
                return PreviousLayer.Gradients;
            }
        }

        public virtual CpuGpuArray Outputs
        {
            get { return GetArray(ArrayName.Outputs); }
        }

        public float[] OutputsGPU
        {
            get { return Outputs.GPUArray; }
        }

        public float[] OutputsCPU
        {
            get { return Outputs.CPUArray; }
        }

        public virtual CpuGpuArray Gradients
        {
            get
            {
                if (!_Arrays.ContainsKey(ArrayName.Gradients)) return null;
                return GetArray(ArrayName.Gradients);
            }
        }

        public float[] GradientsGPU
        {
            get 
            {
                if (Gradients == null) return null;
                return Gradients.GPUArray; 
            }
        }

        public float[] GradientsCPU
        {
            get 
            {
                if (Gradients == null) return null;
                return Gradients.CPUArray; 
            }
        }

        public virtual void CopyToGpu()
        {
            foreach (var array in _Arrays.Values) array.CopyToGpu();
        }

        public virtual void CopyToHost()
        {
            foreach (var array in _Arrays.Values) array.CopyToHost();
        }
        
        public virtual void Calculate(bool train = true)
        {

        }

        public virtual void BackPropagate()
        {

        }

        public virtual void ApplyWeightUpdates(float learnRate, float momentum)
        {
            var minibatchLearnRate = learnRate / this.MinibatchSize;
            var biasLearnRate = learnRate;
            if (this.BiasLearnRate != float.MinValue)
            {
                biasLearnRate = this.BiasLearnRate;
            }
            var minibatchBiasLearnRate = biasLearnRate / (float)this.MinibatchSize;

            var weights = GetArray(ArrayName.Weights);
            if (weights == null) return;
            var weightUpdates = GetArray(ArrayName.WeightUpdates);
            var lastWeightUpdates = GetArray(ArrayName.LastWeightUpdates);
            if (weightUpdates == null) throw new Exception("Weight updates null");
            var biasWeights = GetArray(ArrayName.BiasWeights);
            var biasWeightsUpdates = GetArray(ArrayName.BiasWeightUpdates);
            var lastBiasWeightUpdates = GetArray(ArrayName.LastBiasWeightUpdates);
            if (biasWeightsUpdates == null) throw new Exception("Bias weight updates null");
            if (weights.Length != weightUpdates.Length) throw new Exception("Weights count <> Weightupdates count " + weights.Length + "<>" + weightUpdates.Length);
            if (biasWeights.Length != biasWeightsUpdates.Length) throw new Exception("BiasWeights count <> BiasWeightupdates count " + weights.Length + "<>" + weightUpdates.Length);

            // Add momentun to the weight updates
            var inputLayer = PreviousLayer as DataLayer;
            var mom = momentum;
            var sparseInput = (inputLayer != null) && (inputLayer.IsSparse);
            if (sparseInput)
            {
                momentum = 0f;
            }
            
            if (momentum != 0f)
            {
                this._gpuModule.Blas.Blas.AXPY(momentum , lastWeightUpdates.GPUArray, weightUpdates.GPUArray);
                this._gpuModule.Blas.Blas.AXPY(momentum , lastBiasWeightUpdates.GPUArray, biasWeightsUpdates.GPUArray);
            }

            WeightUpdateCount++;
            if (WeightUpdateCount % RegularizationRatio == 0)
            {
                // Add L2 loss
                if (L2Regularization != 0f)
                {
                    this._gpuModule.Blas.Blas.AXPY(-1f * L2Regularization * (float)RegularizationRatio, weights.GPUArray, weightUpdates.GPUArray);
                    //this._gpuModule.Blas.Blas.AXPY(-1f * biasLearnRate * WeightDecay, biasWeights.GPUArray, biasWeightsUpdates.GPUArray);
                }
            }
            
            this._gpuModule.Blas.Blas.AXPY(minibatchLearnRate, weightUpdates.GPUArray, weights.GPUArray);
            this._gpuModule.Blas.Blas.AXPY(minibatchBiasLearnRate, biasWeightsUpdates.GPUArray, biasWeights.GPUArray); 

            _Arrays[ArrayName.LastWeightUpdates] = weightUpdates;
            _Arrays[ArrayName.WeightUpdates] = lastWeightUpdates;
            _Arrays[ArrayName.LastBiasWeightUpdates] = biasWeightsUpdates;
            _Arrays[ArrayName.BiasWeightUpdates] = lastBiasWeightUpdates;
        }

        public virtual void Free()
        {
            foreach (var key in _Arrays.Keys.ToList())
            {
                var array = _Arrays[key];
                array.Free();
                _Arrays.Remove(key);
            }

            foreach (var key in _IntArrays.Keys.ToList())
            {
                var array = _IntArrays[key];
                array.Free();
                _IntArrays.Remove(key);
            }
        }

        public CpuGpuArray GetArray(ArrayName arrayId)
        {
            if (!_Arrays.ContainsKey(arrayId)) return null;
            var res = _Arrays[arrayId];
            return res;
        }

        public CpuGpuArrayInt GetIntArray(ArrayName arrayId)
        {
            if (!_IntArrays.ContainsKey(arrayId)) return null;
            var res = _IntArrays[arrayId];
            return res;
        }

        public CpuGpuArray AddArray(ArrayName arrayName, int dim)
        {
            var res = new CpuGpuArray(this._gpuModule, dim);
            _Arrays.Add(arrayName, res);
            return res;
        }

        public CpuGpuArray AddArray(ArrayName arrayName, int dim1, int dim2)
        {
            var dim = dim1 * dim2;
            var res = new CpuGpuArray(this._gpuModule, dim1, dim2);
            _Arrays.Add(arrayName, res);
            return res;
        }

        public CpuGpuArrayInt AddIntArray(ArrayName arrayName, int dim)
        {
            var res = new CpuGpuArrayInt(this._gpuModule, dim);
            _IntArrays.Add(arrayName, res);
            return res;
        }

        public CpuGpuArrayInt AddIntArray(ArrayName arrayName, int dim1, int dim2)
        {
            var dim = dim1 * dim2;
            var res = new CpuGpuArrayInt(this._gpuModule, dim1, dim2);
            _IntArrays.Add(arrayName, res);
            return res;
        }

        public virtual string GetTypeDescription()
        {
            var res = this.GetType().ToString().Substring(0,4).ToUpper();
            return res;
        }

        public virtual string GetSizeDescription()
        {
            var res = this.MinibatchSize + "x" + this.Size;
            return res;
        }
    }


}

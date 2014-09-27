using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class FullyConnectedLayer : Layer
    {

        public FullyConnectedLayer(GPUModule gpuModule, Layer previousLayer, int size, string id = "") : base(gpuModule, previousLayer, size, id)
        {
            if (previousLayer != null)
            {
                AddArray(ArrayName.WeightUpdates, InputSize, this.Size);
                AddArray(ArrayName.LastWeightUpdates, InputSize, this.Size);
                AddArray(ArrayName.Weights, InputSize, this.Size);
                AddArray(ArrayName.BiasWeights, this.Size);
                AddArray(ArrayName.BiasWeightUpdates, this.Size);
                AddArray(ArrayName.LastBiasWeightUpdates, this.Size);

                var biasMultipliers = AddArray(ArrayName.BiasMultiplier, this.MinibatchSize, 1);
                biasMultipliers.FillCPU(1f);
            }
        }

        public override void Calculate(bool train = true)
        {
            var inputs = this.Inputs;
            var weights = this.Weights;
            var outputs = this.Outputs;

            var inputLayer = PreviousLayer as DataLayer;
            if ((inputLayer!=null) && (inputLayer.IsSparse))
            {
                _gpuModule.MultiplySparse(inputLayer.SparseMatrix, weights, outputs, transposeA: false);
            }
            else 
            {
            _gpuModule.Blas.GemmRowMajor(inputs, weights, outputs);
            }

            var biases = this.BiasWeights;

            var biasMultiplier = this.BiasMultiplier;
            _gpuModule.Blas.GemmRowMajor(biasMultiplier, biases, outputs, cMultiplier: 1f);
        }

        public override void BackPropagate()
        {            
            // Compute weight updates
            var weightUpdates = this.WeightUpdates;
            var gradients = this.Gradients;
            var inputs = this.Inputs;

            var inputLayer = PreviousLayer as DataLayer;
            if ((inputLayer!=null) && (inputLayer.IsSparse))
            {
                _gpuModule.MultiplySparse(inputLayer.SparseMatrix, gradients, weightUpdates,transposeA: true);
            }
            else
            {
                _gpuModule.Blas.GemmRowMajor(inputs, gradients, weightUpdates, transposeA: true);
            }

            // Todo: use GEMV
            var biasMultiplier = this.BiasMultiplier;
            var biasWeightupdates = this.BiasWeightUpdates;
            _gpuModule.Blas.GemmRowMajor(gradients, biasMultiplier, biasWeightupdates, transposeA: true);

            // Compute layer(N-1) gradients
            var weights = this.Weights;
            var inputGradients = this.InputGradients;
            if (inputGradients != null)
            {
                _gpuModule.Blas.GemmRowMajor(gradients, weights, inputGradients, transposeB: true);
            }
        }

        public CpuGpuArray Weights
        {
            get { return GetArray(ArrayName.Weights); }
        }

        public float[] WeightsGPU
        {
            get { return Weights.GPUArray; }
        }

        public float[] WeightsCPU
        {
            get { return Weights.CPUArray; }
        }

        public CpuGpuArray WeightUpdates 
        { 
            get { return GetArray(ArrayName.WeightUpdates); } 
        }

        public float[] WeightUpdatesGPU
        {
            get { return WeightUpdates.GPUArray; }
        }

        public float[] WeightUpdatesCPU
        {
            get { return WeightUpdates.CPUArray; }
        }

        public CpuGpuArray BiasWeights
        {
            get { return GetArray(ArrayName.BiasWeights); }
        }

        public float[] BiasWeightsGPU
        {
            get { return BiasWeights.GPUArray; }
        }

        public float[] BiasWeightsCPU
        {
            get { return BiasWeights.CPUArray; }
        }

        public CpuGpuArray BiasMultiplier
        {
            get { return GetArray(ArrayName.BiasMultiplier); }
        }

        public CpuGpuArray BiasWeightUpdates
        {
            get { return GetArray(ArrayName.BiasWeightUpdates); }
        }

        public override string GetTypeDescription()
        {
            return "FC";
        }
    }
}

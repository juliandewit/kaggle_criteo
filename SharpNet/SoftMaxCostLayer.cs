using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class SoftMaxCostLayer : CostLayer
    {
        FullyConnectedLayer _fullyConnectedLayer;
        public SoftMaxCostLayer(GPUModule gpuModule, FullyConnectedLayer previousLayer, DataLayer labelLayer, string id = "") : base(gpuModule, previousLayer, labelLayer, 0, id)
        {
            this.Size = previousLayer.Size;
            AddArray(ArrayName.CorrectlyPredictedLabels, MinibatchSize, 1);
            AddArray(ArrayName.Outputs, MinibatchSize, this.Size);
            _fullyConnectedLayer = previousLayer;
        }

        public CpuGpuArray CorrectlyPredictedLabels
        {
            get { return GetArray(ArrayName.CorrectlyPredictedLabels); }
        }

        public CpuGpuArray Labels
        {
            get
            {
                return GroundTruthLayer.Outputs;
            }
        }

        public override float[] GetPredictedLabels(float threshold = 0f)
        {
            Outputs.CopyToHost();
            var res = new float[MinibatchSize];
            for (var minibatchNo = 0; minibatchNo < MinibatchSize; minibatchNo++)
            {
                var maxLabel = float.MinValue;
                var maxLabelChance = float.MinValue;
                for (var labelNo = 0; labelNo < this.Size; labelNo++)
                {
                    var labelChance = Outputs[minibatchNo, labelNo];
                    if (labelChance > maxLabelChance)
                    {
                        maxLabel = labelNo;
                        maxLabelChance = labelChance;
                    }
                    res[minibatchNo] = maxLabel;
                }
            }
            return res;
        }

        public override CpuGpuArray Gradients
        {
            get 
            {
                var res = GetArray(ArrayName.Gradients);
                if (res == null)
                {
                    res = PreviousLayer.Gradients; 
                }
                return res;
            }
        }

        public CpuGpuArray Weights
        {
            get { return _fullyConnectedLayer.Weights; }
        }

        public CpuGpuArray BiasWeights
        {
            get { return _fullyConnectedLayer.BiasWeights; }
        }

        public override void Calculate(bool train = true)
        {
            _gpuModule.CalculateSoftmax(this);
        }

        public override void BackPropagate()
        {
        }

        public override string GetTypeDescription()
        {
            return "SMAX";
        }
    }
}

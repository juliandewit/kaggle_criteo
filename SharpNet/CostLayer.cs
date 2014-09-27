using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{ 
    public class CostLayer : Layer
    {
        public DataLayer GroundTruthLayer;

        public CostLayer(GPUModule gpuModule, Layer previousLayer, DataLayer groundThruthLayer, int size, string id = "") : base(gpuModule, previousLayer, size, id)
        {
            GroundTruthLayer = groundThruthLayer;
        }

        public CpuGpuArray Cost
        {
            get { return GetArray(ArrayName.Cost); }
        }

        public virtual float[] GetPredictedLabels(float threshold = 0f)
        {
            throw new NotImplementedException();
        }

        public virtual float GetLoss()
        {
            throw new NotImplementedException();
        }
    }
}

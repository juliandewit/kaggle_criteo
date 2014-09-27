using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class DropoutLayer : Layer
    {
        public DropoutLayer(GPUModule gpuModule, Layer previousLayer, string id = "") : base(gpuModule, previousLayer, previousLayer.Size, id)
        {
            AddArray(ArrayName.DropoutMask, MinibatchSize, this.Size);
        }

        public override void Calculate(bool train)
        {
            _gpuModule.CalculateDropout(this, train);
        }

        public override void BackPropagate()
        {
            _gpuModule.BackPropagateDropout(this);
        }
    }
}

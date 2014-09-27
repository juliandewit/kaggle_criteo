using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class ActivationLayer : Layer
    {
        public ActivationLayer(GPUModule gpuModule, Layer previousLayer, string id = "", bool createOutputs = true) : base(gpuModule, previousLayer, 0, id)
        {
            if (createOutputs)
            {
                this.Size = previousLayer.Size;
                AddArray(ArrayName.Outputs, previousLayer.MinibatchSize, this.Size);
                AddArray(ArrayName.Gradients, previousLayer.MinibatchSize, this.Size);
            }
        }

        public override CpuGpuArray Outputs
        {
            get 
            { 
                return this.GetArray(ArrayName.Outputs); 
            }
        }

        public override CpuGpuArray Gradients
        {
            get { return this.GetArray(ArrayName.Gradients); }
        }
    }

    public class ReluLayer : ActivationLayer
    {
        public ReluLayer(GPUModule gpuModule, Layer previousLayer, string id = "") : base(gpuModule, previousLayer, id)
        {
        }

        public override void Calculate(bool train = true)
        {
            _gpuModule.CalculateRelu(this);
        }

        public override void BackPropagate()
        {
            _gpuModule.BackPropagateRelu(this);
        }

        public override string GetTypeDescription()
        {
            return "RELU";
        }
    }


    public class TanhLayer : ActivationLayer
    {
        public TanhLayer(GPUModule gpuModule, Layer previousLayer, string id = "") : base(gpuModule, previousLayer, id)
        {
        }

        public override void Calculate(bool train = true)
        {
            _gpuModule.CalculateTanh(this);
        }

        public override void BackPropagate()
        {
            _gpuModule.BackPropagateTanh(this);
        }

        public override string GetTypeDescription()
        {
            return "TANH";
        }
    }

    public class MaxoutLayer : ActivationLayer
    {
        public int GroupSize = 2;

        public MaxoutLayer(GPUModule gpuModule, Layer previousLayer, int groupSize = 2, string id = "") : base(gpuModule, previousLayer, id, createOutputs: false)
        {
            GroupSize = groupSize;
            if (previousLayer.Size % GroupSize != 0) throw new ArgumentException("Invalid groupsize");
            this.Size = previousLayer.Size / GroupSize;
            AddArray(ArrayName.Outputs, previousLayer.MinibatchSize, this.Size);
            AddArray(ArrayName.Gradients, previousLayer.MinibatchSize, this.Size);
            AddIntArray(ArrayName.Winners, previousLayer.MinibatchSize, this.Size);
        }

        public CpuGpuArrayInt Winners
        {
            get
            {
                var res = GetIntArray(ArrayName.Winners);
                return res;
            }
        }

        public override void Calculate(bool train = true)
        {
            _gpuModule.CalculateMaxout(this);
        }

        public override void BackPropagate()
        {
            _gpuModule.BackPropagateMaxout(this);
        }

        public override string GetTypeDescription()
        {
            return "MAXOUT";
        }


    }
}

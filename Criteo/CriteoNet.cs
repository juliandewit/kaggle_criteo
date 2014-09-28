using SharpNet;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class CriteoNet
    {
        public static Network CreateNetworkRelu(GPUModule module, int minibatchSize)
        {
            var net = new Network(module, minibatchSize: minibatchSize);
            net.AddInputLayer(Constants.TOTAL_VALUE_COUNT, sparseDataSize: minibatchSize * RawRecord.FEATURE_COUNT * 2);
            net.AddLabelLayer(1);
            var fc1 = net.AddFullyConnectedLayer(128);
            fc1.Weights.InitValuesUniformCPU(0.1f);

            fc1.L2Regularization = 0.00001f;
            fc1.RegularizationRatio = 10;
            net.AddReluLayer("REL1");

            var fc2 = net.AddFullyConnectedLayer(256);
            fc2.Weights.InitValuesUniformCPU(0.1f);
            net.AddReluLayer("REL2");
            net.AddDropoutLayer();

            var sm = net.AddSoftmaxLayer(2);
            sm.Weights.InitValuesUniformCPU(0.1f);
            return net;
        }

        public static Network CreateNetworkMaxout(GPUModule module, int minibatchSize)
        {
            var net = new Network(module, minibatchSize: minibatchSize);
            net.AddInputLayer(Constants.TOTAL_VALUE_COUNT, sparseDataSize: minibatchSize * RawRecord.FEATURE_COUNT * 2);
            net.AddLabelLayer(1);
            
            var fc1 = net.AddFullyConnectedLayer(128);
            fc1.Weights.InitValuesUniformCPU(0.1f);

            fc1.L2Regularization = 0.00001f;
            fc1.RegularizationRatio = 10;
            net.AddMaxoutLayer("MAXOUT1", groupsize: 4);

            var fc2 = net.AddFullyConnectedLayer(256);
            fc2.Weights.InitValuesUniformCPU(0.1f);
            net.AddMaxoutLayer("MAXOUT2",groupsize: 2);
            net.AddDropoutLayer();
            
            var sm = net.AddSoftmaxLayer(2);
            sm.Weights.InitValuesUniformCPU(0.1f);
            return net;
        }

    }
}

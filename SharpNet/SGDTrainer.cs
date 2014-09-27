using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class SGDTrainer
    {
        protected Network _network;
        protected DataProvider _trainDataProvider;
        protected DataProvider _testDataProvider;

        public SGDTrainer(Network net,DataProvider trainDataProvider, DataProvider testDataProvider)
        {
            _network = net;
            _testDataProvider = testDataProvider;
            _trainDataProvider = trainDataProvider;
        }

        public virtual void Train(float learnRate, float momentum, int epochs, int epochsBeforeHoldoutTest, int epochsBeforeReport)
        {
            _network.CopyToGpu();
            var batch = _trainDataProvider.GetNextBatch();
            var epochNo = batch.EpocNo;
            var batchNo = batch.BatchNo;
            while (epochNo < epochs)
            {
                if (((epochNo % epochsBeforeHoldoutTest == 0) && batchNo == 1))
                {
                    DoTest(epochNo, batchNo);
                }
                
                _network.InputLayer.SetData(batch.Inputs, 0f, 0.4f);
                _network.LabelLayer.SetData(batch.Labels);
                _network.Calculate(train: true);
                _network.BackPropagate();

              
                _network.ApplyWeightUpdates(learnRate, momentum);

                if (((epochNo % epochsBeforeReport == 0) && batchNo == 1))
                {

                    var costLayer = _network.CostLayer as SoftMaxCostLayer;
                    if (costLayer != null)
                    {
                        costLayer.CorrectlyPredictedLabels.CopyToHost();
                        var sum = costLayer.CorrectlyPredictedLabels.CPUArray.Sum();
                        var correctPercentage = (sum * 100f) / (float)costLayer.CorrectlyPredictedLabels.CPUArray.GetLength(0);
                        var fNo = (epochNo + "." + batchNo).PadRight(10);
                        Console.WriteLine(fNo + " correct% : " + correctPercentage);
                    }
                }

                batch = _trainDataProvider.GetNextBatch();
                epochNo = batch.EpocNo;
                batchNo = batch.BatchNo;
            }
        }

        public virtual void DoTest(int epochNo, int batchNo)
        {
            var testBatch = _testDataProvider.GetNextBatch();
            _network.InputLayer.SetData(testBatch.Inputs);
            _network.InputLayer.CopyToGpu();
            _network.LabelLayer.SetData(testBatch.Labels);
            _network.LabelLayer.CopyToGpu();
            _network.Calculate(train: false);
            _network.CopyToHost();

            var costLayer = _network.CostLayer as SoftMaxCostLayer;
            var sum = 0f;
            testBatch.PredictedLabels = costLayer.GetPredictedLabels();
            var correctPercentage = _testDataProvider.GetCorrectLabelPercentage();
            var fNo = (epochNo + "." + batchNo).PadRight(10);
            Console.WriteLine("Test " + fNo + " correct : " + correctPercentage);
            _network.CopyToHost();

        }

    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cudafy.Maths.BLAS;
using System.IO;
using System.Diagnostics;

namespace SharpNet
{
    public class Network
    {
        int _minibatchSize = 128;
        GPUModule _gpuModule;
        public DataLayer InputLayer;
        public DataLayer LabelLayer;
        public CostLayer CostLayer;
        public List<Layer> Layers = new List<Layer>();
        public DataProvider TrainDataProvider;
        public DataProvider TestDataProvider;

        public List<float> TestLosses = new List<float>();
        public List<float> TrainLosses = new List<float>();
        List<float> _movingAverageTestLosses = new List<float>();
        List<float> _movingAverageTrainLosses = new List<float>();
        public int SamplesPerMovingAverage = 1;
        List<float> TrainRecordsPerSecond = new List<float>();
        public int TrainRecordsCount = 0;
        public Stopwatch Timer;

        public Network(GPUModule gpuModule, int minibatchSize = 128)
        {
            _minibatchSize = minibatchSize;
            _gpuModule = gpuModule;
            Timer = Stopwatch.StartNew();
        }


        int tmp = 0;
        public void Calculate(bool train)
        {
            var provider = TrainDataProvider;
            if (!train) provider = TestDataProvider;
            if (provider != null)
            {
                var nextBatch = provider.GetNextBatch();
                if (!nextBatch.IsGPUData)
                {
                    if (!InputLayer.IsSparse)
                    {
                        InputLayer.SetData(nextBatch.Inputs);
                    }
                    else
                    {
                        InputLayer.SetSparseData(nextBatch.SparseValues.ToList(), nextBatch.SparseIndices.ToList());
                    }
                    LabelLayer.SetData(nextBatch.Labels);
                }
                else
                {
                    InputLayer.SetGPUData(nextBatch.GpuInputs);
                    LabelLayer.SetGPUData(nextBatch.GpuLabels);
                }
                tmp++;
            }

            foreach (var layer in Layers)
            {
                //if (layer is Conv1DLayer) continue;
                layer.Calculate(train);
            }
            InputLayer.IncBatchNo();
            LabelLayer.IncBatchNo();
            if (train) TrainRecordsCount += _minibatchSize;

        }

        public void BackPropagate()
        {
            foreach (var layer in Layers.ToArray().Reverse())
            {
                layer.BackPropagate();
            }

        }

        public void ApplyWeightUpdates(float learnRate, float momentum)
        {
            foreach (var layer in Layers.ToArray().Reverse())
            {
                layer.ApplyWeightUpdates(learnRate, momentum);
            }
        }

        public void CopyToGpu()
        {
            foreach (var layer in Layers)
            {
                layer.CopyToGpu();
            }
        }

        public void CopyToHost()
        {
            foreach (var layer in Layers)
            {
                layer.CopyToHost();
            }
        }

        public void Free()
        {
            foreach (var layer in Layers)
            {
                layer.Free();
            }
        }

        public DataLayer AddInputLayer(int size, int minibatchSize = 128, int batchesPerLoad = 1, int sparseDataSize = 0)
        {
            if (Layers.Count != 0) throw new Exception("There are already layers in the network");
            var layer = new DataLayer(_gpuModule, size:size, batchesPerLoad:batchesPerLoad, miniBatchSize:this._minibatchSize, sparseDataSize: sparseDataSize);
            Layers.Add(layer);
            InputLayer = layer;
            return layer;
        }

        public DataLayer AddLabelLayer(int size, int minibatchSize = 128, int batchesPerLoad = 1)
        {
            LabelLayer = new DataLayer(_gpuModule, size, batchesPerLoad: batchesPerLoad, miniBatchSize: this._minibatchSize);
            Layers.Insert(0, LabelLayer);
            return LabelLayer;
        }

        public FullyConnectedLayer AddFullyConnectedLayer(int size, string id = "")
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            var fcLayer = new FullyConnectedLayer(_gpuModule, lastLayer, size, id: id);
            Console.WriteLine("FC layer : " + fcLayer.Size + " weights : " + fcLayer.WeightsCPU.Length);
            Layers.Add(fcLayer);
            return fcLayer;
        }

        public ReluLayer AddReluLayer(string id = "")
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            var reluLayer = new ReluLayer(_gpuModule, lastLayer, id: id );
            Layers.Add(reluLayer);
            return reluLayer;
        }

        public TanhLayer AddTanhLayer(string id = "")
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            var tanhLayer = new TanhLayer(_gpuModule, lastLayer, id: id);
            Layers.Add(tanhLayer);
            return tanhLayer;
        }

        public MaxoutLayer AddMaxoutLayer(string id = "", int groupsize = 2)
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            var maxoutLayer = new MaxoutLayer(_gpuModule, lastLayer, id: id, groupSize: groupsize);
            Layers.Add(maxoutLayer);
            return maxoutLayer;
        }
        
        public DropoutLayer AddDropoutLayer()
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            //var convLayer = lastLayer as Conv1DLayer;
            //if (convLayer == null) throw new Exception("Last layer is not a conv1d layer");
            var layer = new DropoutLayer(_gpuModule, lastLayer);
            Layers.Add(layer);
            Console.WriteLine("Dropout layer.. ");
            return layer;
        }

        public SoftMaxCostLayer AddSoftmaxLayer(int size, string id = "")
        {
            var lastLayer = Layers.Last();
            if (lastLayer == null) throw new Exception("There must be one or more layers in the network");
            if (CostLayer != null) throw new Exception("There is already a cost layer");

            var fcLayer = new FullyConnectedLayer(_gpuModule, lastLayer, size, id: id + "_fc");
            Layers.Add(fcLayer);
            var smLayer = new SoftMaxCostLayer(_gpuModule, fcLayer, this.LabelLayer, id: id);
            Layers.Add(smLayer);
            CostLayer = smLayer;
            return smLayer;
        }
    
        public Layer GetLayerById(string id)
        {
            var res = Layers.SingleOrDefault(x => x.Id == id);
            return res;
        }

        public void SaveWeightsAndParams(string dir, string name)
        {
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, name + ".xml");
            SaveWeightsAndParams(path);
        }

        public void SaveWeightsAndParams(string path)
        {
            NetSerializer.SerializeNet(path, this);
        }

        public void LoadStructureWeightsAndParams(string dir, string name)
        {
            var path = Path.Combine(dir, name + ".xml");
            LoadStructureWeightsAndParams(path);
        }
        
        public void LoadStructureWeightsAndParams(string path)
        {
            if (!File.Exists(path)) throw new Exception("Could not find network file " + path);
            NetSerializer.DeserializeNet(path, this);
            this.CopyToGpu();
        }

        public void RegisterTestLoss(long time, float loss)
        {
            RegisterLoss(time, loss, train:false);
        }

        public void RegisterTrainLoss(long time, float loss)
        {
            RegisterLoss(time, loss, train:true);
        }

        public void RegisterLoss(long time, float loss, bool train = true, string savePathWhenBest = null)
        {
            var movingAverageList = (train) ? _movingAverageTrainLosses : _movingAverageTestLosses;
            var lossList = (train) ? TrainLosses : TestLosses;
            if (movingAverageList.Count >= SamplesPerMovingAverage)
            {
                var currentLoss = movingAverageList.Average();
                lossList.Add(currentLoss);
                movingAverageList.Clear();

                if ((!train) && (savePathWhenBest != null))
                {
                    var min = lossList.Min();
                    if (min == currentLoss)
                    {
                        Console.WriteLine("New best : " + min + "saving..");
                        var bestLossTxt = ((int)Math.Round(min * 1000)).ToString();
                        this.SaveWeightsAndParams(savePathWhenBest.Replace("XX", bestLossTxt));
                    }
                }
            }
            movingAverageList.Add(loss);
        }

        public long GetTrainRecordsPerSecond()
        {
            var res = (long)TrainRecordsCount / (Timer.ElapsedMilliseconds/1000);
            return res;
        }
    }
    }

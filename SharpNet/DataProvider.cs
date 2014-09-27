using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class DataProvider
    {
        public List<DataBatch> Batches = new List<DataBatch>();
        public DataBatch CurrentBatch;
        public int _currentBatch = 0;
        public int _currentEpoch = 0;

        public virtual DataBatch GetNextBatch()
        {
            _currentBatch++;
            if (_currentBatch > Batches.Count)
            {
                _currentBatch = 1;
                _currentEpoch++;
            }
            var batch = Batches[_currentBatch-1];
            batch.EpocNo = _currentEpoch;
            batch.BatchNo = _currentBatch;
            CurrentBatch = batch;
            return batch;
        }

        public float GetCorrectLabelPercentage()
        {
            var totalLabelCount = 0;
            var totalCorrect = 0;
            foreach (var batch in Batches)
            {
                if (batch.PredictedLabels != null)
                {
                    totalLabelCount += batch.Labels.Length;
                    totalCorrect += batch.GetCorrectPredictedCount();
                }
            }
            var res = (float)totalCorrect / (float)totalLabelCount;
            return res;
        }
    }

    public class DataBatch
    {
        public bool IsGPUData = false;
        public bool IsSparse = false;

        public int MinibatchSize = 128;
        public int InputSize = 0;
        public int LabelSize = 0;
        public float[] Inputs;
        public float[] Labels;
        public float[] LabelWeights;
        public float[] Targets;
        float[] _predictedLabels;
        public object[] TagData;
        public float[] GpuInputs;
        public float[] GpuLabels;

        public int[][] SparseIndices;
        public float[][] SparseValues;

        public float[] PredictedLabels
        {
            get
            {
                if (SourceBatch != null) return SourceBatch.PredictedLabels;
                return _predictedLabels;
            }
            set
            {
                if (SourceBatch != null)
                {
                    SourceBatch.PredictedLabels = value;
                }
                else
                {
                    _predictedLabels = value;
                }
            }
        }


        public int BatchNo = 0;
        public int EpocNo = 0;
        public DataBatch SourceBatch;

        public DataBatch(int inputSize,int labelsSize, int minibatchSize = 128, bool sparse = false) 
        {
            InputSize = inputSize;
            LabelSize = labelsSize;
            MinibatchSize = minibatchSize;
            IsSparse = sparse;
            if (!IsSparse)
            {
                Inputs = new float[MinibatchSize * InputSize];
            }
            else
            {
                SparseIndices = new int[MinibatchSize][];
                SparseValues = new float[MinibatchSize][];
            }

            Labels = new float[MinibatchSize * LabelSize];

            //PredictedLabels = new float[MinibatchSize * LabelSize];
            TagData = new object[MinibatchSize];
        }

        public int GetCorrectPredictedCount()
        {
            var res = 0;
            for (var i = 0; i < Labels.Length; i++)
            {
                if (PredictedLabels[i] == Labels[i]) res++;
            }
            return res;
        }

        public void CopyToHost(GPUModule gpuModule)
        {
            if (!this.IsGPUData) throw new Exception("Not gpu anabled");
            gpuModule.Gpu.CopyFromDevice(this.GpuInputs, this.Inputs);
            gpuModule.Gpu.CopyFromDevice(this.GpuLabels, this.Labels);
        }
    }
}

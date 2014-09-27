using SharpNet;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class OneHotRecordProvider : DataProvider
    {
        public string Id { get; set; }
        public int CurrentSet = 0;
        public int CurrentEpochBatch = 0;
        int _totalBatchCount = 0;
        int _batchesRead = 0;
        bool _shuffle = false;
        public List<OneHotRecordReadOnly> _records;

        public GPUModule GpuModule;
        public bool BalanceClasses = false;
        int _currentSet = 0;


        public OneHotRecordProvider(GPUModule gpuModule, List<OneHotRecordReadOnly> records, string id = "", bool shuffleEveryEpoch = false)
        {
            Id = id;
            _records = records;
            GpuModule = gpuModule;
            _shuffle = shuffleEveryEpoch;
            if (_shuffle) _records.Shuffle();
        }

        public override DataBatch GetNextBatch()
        {
            if (_currentBatch >= Batches.Count)
            {
                LoadNextBatchSet();
                _currentBatch = 0;
            }
            var batch = Batches[_currentBatch];
            CurrentEpochBatch++;
            batch.EpocNo = _currentEpoch;
            batch.BatchNo = CurrentEpochBatch; 
            CurrentBatch = batch;
            _currentBatch++;
            return batch; 
        }

        bool _loaded = false;
        int _recNo = 0;
        public void LoadNextBatchSet()
        {
            if (_currentEpoch >= 0)
            {
                for (int setIdx = 0; setIdx < Constants.BATCHES_PER_SET; setIdx++)
                {
                    DataBatch batch;
                    if ((Batches == null) || (Batches.Count < (Constants.BATCHES_PER_SET)))
                    {
                        batch = new DataBatch(Constants.TOTAL_VALUE_COUNT, 1, sparse: true);
                        Batches.Add(batch);
                    }
                    else
                    {
                        batch = Batches[setIdx];
                    }

                    for (var minibatchIdx = 0; minibatchIdx < Constants.MINIBATCH_SIZE; minibatchIdx++)
                    {
                        OneHotRecordReadOnly rec = _records[_recNo];
                        rec.CopyDataToSparseArray(batch.SparseIndices, batch.SparseValues, minibatchIdx);
                        batch.Labels[minibatchIdx] = rec.Label;
                        _recNo++;
                        if (_recNo >= _records.Count)
                        {
                            _recNo = 0;
                            if (_totalBatchCount == 0) _totalBatchCount = _batchesRead;
                            _batchesRead = 0;
                            _currentEpoch++;
                            CurrentEpochBatch = 0;
                            if (_shuffle) _records.Shuffle();
                        }
                        _batchesRead++;
                    }
                }
            }
           
            _currentBatch = 0;
            CurrentSet++;
            _loaded = true;
        }

    }
}

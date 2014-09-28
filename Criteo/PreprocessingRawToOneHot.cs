using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class PreprocessingRawToOneHot
    {
        static Dictionary<int, Dictionary<int, int>> _categoricalIdices;
        static Dictionary<string, int> _hashIndices;

        public static void ConvertRawToOneHot(string rawTrainBinSrcPath, string rawTestBinSrcPath, string oneHotTrainDstPath, string oneHotTestDstPath, bool encodeMissingValues = true, int encodeTestNotrainAs = Constants.VALUE_MISSING, bool logTransformNumericValues = false)
        {
            _hashIndices = new Dictionary<string, int>();
            _categoricalIdices = new Dictionary<int, Dictionary<int, int>>();
            ConvertRawToOneHot(rawTrainBinSrcPath, oneHotTrainDstPath, false, encodeMissingValues, encodeTestNotrainAs, logTransformNumericValues);
            ConvertRawToOneHot(rawTestBinSrcPath, oneHotTestDstPath, true, encodeMissingValues, encodeTestNotrainAs, logTransformNumericValues);
        }

        protected static void ConvertRawToOneHot(string rawSrcPath, string oneHotDstPath, bool isTestSet, bool encodeMissingValues = true, int encodeTestNotrainAs = Constants.VALUE_MISSING, bool logTransformNumericValues = false)
        {
            _hashIndices = new Dictionary<string, int>();
            _categoricalIdices = new Dictionary<int, Dictionary<int, int>>();

            if (File.Exists(oneHotDstPath)) File.Delete(oneHotDstPath);
            var stream = File.OpenWrite(oneHotDstPath);
            var writer = new BinaryWriter(stream);

            var label = (isTestSet) ? "test" : "train";
            Console.WriteLine("Converting " + label + " records");

            var recNo = 0;
            foreach (var raw in RawRecord.EnumerateBinLines(rawSrcPath))
            {
                var click = ConvertRecord(raw, recNo, isTestSet);
                click.WriteBinary(writer);
                recNo++;
            }

            writer.Flush();
            stream.Close();
        }

        
        public static OneHotRecord ConvertRecord(RawRecord raw, int recordIndex, bool isTestSet, bool encodeMissingValues = true, int encodeTestNotrainAs = Constants.VALUE_MISSING, bool logTransformNumericValues = false)
        {
            var res = new OneHotRecord();
            res.Label = raw.Label;
            res.Id = raw.Id;
            for (short i = 0; i < RawRecord.NUMERIC_COUNT; i++)
            {
                var colNo = i + 1;
                var val = raw.Values[i];
                if (val == Int32.MinValue) 
                {
                    res.NumericData[i] = float.NaN;
                    // Register N/A
                    res.SetNA(i);
                }
                else
                {
                    if (val != 0)
                    {
                        if (logTransformNumericValues)
                        {
                            val += 2;
                            if (colNo == 2) val += 2;
                            val = (int)(Math.Log(val) * 100d);
                        }
                    }
                    res.NumericData[i] = val;
                }
            }

            bool isNew = false;
            for (short catNo = 0; catNo < RawRecord.CATEGORICAL_COUNT; catNo++ )
            {
                var rawVal = raw.Values[RawRecord.NUMERIC_COUNT + catNo];

                // Recode testnottrain
                if (rawVal == Constants.VALUE_TESTNOTTRAIN) rawVal = encodeTestNotrainAs;
                // Skip missing values ?
                if ((rawVal == Constants.VALUE_MISSING) && (!encodeMissingValues)) continue;

                if (_categoricalIdices.ContainsKey(catNo + 1))
                {
                    var catVal = GetCategorical(catNo + 1, rawVal, out isNew);
                    res.SetCategorical(catNo + 1, catVal);
                }
                else
                {
                    // Hashing trick
                    var hash = GetMurmurHash(catNo, rawVal);
                    sbyte value = 1;
                    if (hash < 0)
                    {
                        value = -1;
                        hash = -hash;
                    }
                    var hashIndex = hash % Constants.HASH_SPACE_SIZE;
                    res.StoreHashedValue(hashIndex,value);
                }
            }

            return res;
        }

        protected static int GetMurmurHash(int cat, int val)
        {
            var key = cat + "_" + val;
            var res = 0;
            if (!_hashIndices.TryGetValue(key, out res))
            {
                res = MurMur3.Hash(key);
                _hashIndices[key] = res;
            }
            return res;
        }

        protected static int GetCategorical(int cat, int value, out bool isNew)
        {
            isNew = false;
            var index = _categoricalIdices[cat];
            if (index.ContainsKey(value)) return index[value];
            var newVal = 0;
            newVal = index.Count;
            index[value] = newVal;
            isNew = true;
            return newVal;
        }



    }
}

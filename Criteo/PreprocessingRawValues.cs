using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class PreprocessingRawValues
    {
        static Dictionary<int, int>[] _trainCounts;
        static Dictionary<int, int>[] _testCounts;
        static Dictionary<int, int> _categoricalValueFrequencyFilter;

        public static void ConvertCSVToBinary(string csvPath, string binaryPath)
        {
            Console.WriteLine("Converting CSV to binary");
            if (File.Exists(binaryPath)) File.Delete(binaryPath);
            var fileStream = File.OpenWrite(binaryPath);
            var deflateStream = new DeflateStream(fileStream, CompressionMode.Compress);
            var writer = new BinaryWriter(deflateStream);

            foreach (var rawRecord in RawRecord.EnumerateCSVFile(csvPath))
            {
                rawRecord.WriteBinary(writer);
            }

            writer.Flush();
            deflateStream.Flush();
            deflateStream.Close();
            fileStream.Close();
        }

        public static void RecodeCategoricalValues(string binTrainPath, string binTestPath, string recodedTrainPath, string recodedTestPath, Dictionary<int, int> categoricalValueFrequencyFilter)
        {
            Console.WriteLine("Recoding featurevalues ");
            _categoricalValueFrequencyFilter = categoricalValueFrequencyFilter;

            CountFeatures(binTrainPath, false);
            CountFeatures(binTestPath, true);

            Console.WriteLine("Cleaning..");
            Process(binTrainPath, recodedTrainPath, false);
            Process(binTestPath, recodedTestPath, false);
            foreach (var x in _trainCounts) x.Clear();
            foreach (var x in _testCounts) x.Clear();

            CountFeatures(recodedTrainPath, false, checkValues: false);
            CountFeatures(recodedTestPath, true, checkValues: false);
            Console.WriteLine("Done..");
        }

        protected static void Process(string srcPath, string dstPath, bool test)
        {
            if (File.Exists(dstPath)) File.Delete(dstPath);
            var fileStream = File.OpenWrite(dstPath);
            var deflateStream = new DeflateStream(fileStream, CompressionMode.Compress);
            var writer = new BinaryWriter(deflateStream);
            foreach (var rec in RawRecord.EnumerateBinLines(srcPath))
            {
                for (var i = 0; i < RawRecord.CATEGORICAL_COUNT; i++)
                {
                    var catNo = i + 1;
                    var idx = RawRecord.NUMERIC_COUNT + i;
                    var val = rec.Values[idx];
                    var testCount = _testCounts[catNo][val];
                    var trainCount = _trainCounts[catNo][val];


                    if (testCount == 0)
                    {
                        rec.Values[idx] = Constants.VALUE_TRAINNOTTEST;
                        continue;
                    }
                    if (trainCount == 0)
                    {
                        rec.Values[idx] = Constants.VALUE_TESTNOTTRAIN;
                        //rec.Values[idx] = RawRecord.MISSING;
                        continue;
                    }

                    var threshHold = _categoricalValueFrequencyFilter[catNo];
                    if (trainCount < threshHold)
                    {
                        rec.Values[idx] = Constants.VALUE_LOWFREQUENCY;
                    }
                }
                rec.WriteBinary(writer);
            }
            writer.Flush();
            deflateStream.Flush();
            deflateStream.Close();
            fileStream.Close();
        }

        public static void CountFeatures(string path, bool test, bool checkValues = true)
        {
            Console.WriteLine("Counting features.." + ((test) ? "test" : "train"));
            if (_trainCounts == null)
            {
                _trainCounts = new Dictionary<int, int>[RawRecord.CATEGORICAL_COUNT + 1];
                _testCounts = new Dictionary<int, int>[RawRecord.CATEGORICAL_COUNT + 1];
                for (var i = 0; i < RawRecord.CATEGORICAL_COUNT + 1; i++)
                {
                    _trainCounts[i] = new Dictionary<int, int>();
                    _testCounts[i] = new Dictionary<int, int>();
                }
            }
            var recNo = 0;
            foreach (var rawLine in RawRecord.EnumerateBinLines(path))
            {
                for (var i = 0; i < RawRecord.CATEGORICAL_COUNT; i++)
                {
                    var catNo = i + 1;
                    var val = rawLine.Values[RawRecord.NUMERIC_COUNT + i];
                    IncFeature(_trainCounts[catNo], _testCounts[catNo], val, test, checkValues: checkValues);
                }
                recNo++;
            }

            var counts = _trainCounts;
            if (test) counts = _testCounts;
            for (var i = 0; i < RawRecord.CATEGORICAL_COUNT; i++)
            {
                var catNo = i + 1;
                Console.WriteLine("CAT : " + catNo + " : " + counts[catNo].Count);
                Console.WriteLine("  MISSING : " + GetCount(counts, catNo, Constants.VALUE_MISSING));
                Console.WriteLine("  TESTNOTTRAIN : " + GetCount(counts, catNo, Constants.VALUE_TESTNOTTRAIN));
                Console.WriteLine("  TOOLOWCOUNT : " + GetCount(counts, catNo, Constants.VALUE_LOWFREQUENCY));
                Console.WriteLine("  TRAINNOTTEST : " + GetCount(counts, catNo, Constants.VALUE_TRAINNOTTEST));
            }
            Console.WriteLine("Total : " + " : " + counts.Sum(x => x.Count));
        }

        protected static void IncFeature(Dictionary<int, int> trainCounts, Dictionary<int, int> testCounts, int featureValue, bool test, bool checkValues = false)
        {

            if (checkValues)
            {
                if (featureValue == Constants.VALUE_TRAINNOTTEST) throw new ArgumentException();
                if (featureValue == Constants.VALUE_TESTNOTTRAIN) throw new ArgumentException();
                if (featureValue == Constants.VALUE_LOWFREQUENCY) throw new ArgumentException();
            }
            if (!trainCounts.ContainsKey(featureValue)) trainCounts[featureValue] = 0;
            if (!testCounts.ContainsKey(featureValue)) testCounts[featureValue] = 0;
            if (test)
            {
                testCounts[featureValue]++;
            }
            else
            {
                trainCounts[featureValue]++;
            }
        }

        protected static int GetCount(Dictionary<int, int>[] counts, int catNo, int featureValue)
        {
            var res = 0;
            var catNoCounts = counts[catNo];
            if (catNoCounts.ContainsKey(featureValue)) res = catNoCounts[featureValue];
            return res;
        }

    }
}

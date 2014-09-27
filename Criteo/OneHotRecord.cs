using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class OneHotRecord
    {
        public int Id;
        public int Label;
        public float[] NumericData;
        public Dictionary<int, sbyte> SparseData;

        public OneHotRecord()
        {
            NumericData = new float[RawRecord.NUMERIC_COUNT];
            SparseData = new Dictionary<int, sbyte>();
        }

        public OneHotRecord(BinaryReader reader) : this()
        {
            Id = reader.ReadInt32();
            Label = reader.ReadInt32();
            var floatBytes = reader.ReadBytes(NumericData.Length * 4);
            Buffer.BlockCopy(floatBytes, 0, NumericData, 0, NumericData.Length * 4);
            var valCount = reader.ReadInt32();
            for (var i = 0; i < valCount; i++)
            {
                var key = reader.ReadInt32();
                var value = reader.ReadSByte();
                SparseData[key] = value;
            }
        }

        public void SetCategorical(int catIdx, int valueIndex)
        {
            SetCategorical(Constants.ONEHOT_FEATURE_START_INDICES[catIdx], Constants.ONEHOT_FEATURE_VALUE_COUNTS[catIdx], valueIndex);
        }

        public void SetCategorical(int baseIndex, int size, int valueIndex)
        {
            if (size <= valueIndex) throw new Exception("Index to large");
            var index = baseIndex + valueIndex;
            SetSparseValue(index, (sbyte)1);
        }

        public void StoreHashedValue(int index, sbyte value)
        {
            AddSparseValue(Constants.HASH_SPACE_START_INDEX + index, value);
        }

        public void SetNA(int index)
        {
            SetSparseValue(Constants.NUMERICMISSING_START_INDEX + index, (sbyte)1);
        }

        public void SetSparseValue(int index, sbyte value)
        {
            if (index > Constants.TOTAL_VALUE_COUNT) throw new Exception("Index to large");
            SparseData[index] = value;
        }

        public void AddSparseValue(int index, sbyte value)
        {
            sbyte existing = 0;
            if (!SparseData.TryGetValue(index, out existing))
            {
                SparseData[index] = value;
                return;
            }
            existing += value;
            SparseData[index] = existing;
        }

        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Id);
            writer.Write(Label);
            var bytes = new byte[RawRecord.NUMERIC_COUNT * 4];
            Buffer.BlockCopy(NumericData, 0, bytes, 0, RawRecord.NUMERIC_COUNT * 4);
            writer.Write(bytes);
            writer.Write(SparseData.Count);
            foreach (var sparseItem in SparseData)
            {
                writer.Write(sparseItem.Key);
                writer.Write(sparseItem.Value);
            }
        }

        public static IEnumerable<OneHotRecord> EnumerateBinLines(string path)
        {
            var fileStream = File.OpenRead(path);
            var reader = new BinaryReader(fileStream);
            var lineNo = 0;
            var quit = false;
            OneHotRecord rec;
            while (!quit)
            {
                lineNo++;
                try
                {
                    rec = new OneHotRecord(reader);
                }
                catch (EndOfStreamException ex)
                {
                    rec = null;
                }
                if (rec == null) break;

                yield return rec;
                if (lineNo % 1000000 == 0) Console.WriteLine("Line :  " + lineNo);
            }
            fileStream.Close();
        }
    
        public static float[] GetMeans(params string[] srcPaths)
        {
            var means = new float[RawRecord.NUMERIC_COUNT];
            var totals = new double[RawRecord.NUMERIC_COUNT];
            var counts = new int[RawRecord.NUMERIC_COUNT];
            var label1count = 0;
            var recordCount = 0;
            foreach (var srcPath in srcPaths)
            {
                foreach (var src in OneHotRecord.EnumerateBinLines(srcPath))
                {
                    if (src.Label != 0) label1count++;
                    recordCount++;
                    for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++)
                    {
                        var val = src.NumericData[i];
                        if (!float.IsNaN(val))
                        {
                            totals[i] += val;
                            counts[i] += 1;
                        }
                    }
                }
            }

            Console.WriteLine("Labels : " + label1count + "//" + recordCount);
            for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++) means[i] = (float)totals[i] / (float)counts[i];
            return means;
        }
        
        public static float[] GetStdDevs(float[] means, params string[] srcPaths)
        {
            var counts = new int[RawRecord.NUMERIC_COUNT];
            var squareds = new double[RawRecord.NUMERIC_COUNT];
            var res = new float[RawRecord.NUMERIC_COUNT];
            foreach (var srcPath in srcPaths)
            {
                foreach (var src in OneHotRecord.EnumerateBinLines(srcPath))
                {
                    for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++)
                    {
                        var val = (double)src.NumericData[i];
                        if (!double.IsNaN(val))
                        {
                            var diff = val - (double)means[i];
                            var squared = diff * diff;
                            squareds[i] += squared;
                            counts[i] += 1;
                        }
                    }
                }
            }
            for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++)
            {
                var squared = squareds[i];
                var count = counts[i];
                res[i] = (float)Math.Sqrt(squared / (double)count);
            }
            return res;
        }
    }
}

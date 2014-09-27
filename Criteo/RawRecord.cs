using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class RawRecord
    {
        // Feature counts
        public const int NUMERIC_COUNT = 13;
        public const int CATEGORICAL_COUNT = 26;
        public static int FEATURE_COUNT = NUMERIC_COUNT + CATEGORICAL_COUNT;

        public int Id;
        public int Label;
        public int[] Values;

        public RawRecord(string csvString)
        {
            Values = new int[FEATURE_COUNT];
            var split = csvString.Split(',');
            var test = split.Length == 40;
            Id = Int32.Parse(split[0]);
            Label = 0;
            if (!test) Label = Int32.Parse(split[1]);
            var startIdx = 1;
            if (!test) startIdx++;
            for (var i = 0; i < NUMERIC_COUNT; i++)
            {
                var val = split[startIdx + i];
                if (!String.IsNullOrEmpty(val))
                {
                    Values[i] = Int32.Parse(val);
                }
                else
                {
                    Values[i] = Int32.MinValue;
                }
            }
            for (var i = 0; i < CATEGORICAL_COUNT; i++)
            {
                var val = split[startIdx + NUMERIC_COUNT + i];
                if (!String.IsNullOrEmpty(val))
                {
                    var tmp = Int32.Parse(val, NumberStyles.HexNumber, null);;
                    Values[i + NUMERIC_COUNT] = tmp;
                }
                else
                {
                    Values[i + NUMERIC_COUNT] = Int32.MinValue;
                }
            }
        }

        public static IEnumerable<RawRecord> EnumerateCSVFile(string path)
        {
            var lineNo = 0;
            foreach (var line in File.ReadLines(path))
            {
                lineNo++;
                if (lineNo == 1) continue;
                var rec = new RawRecord(line);

                if (lineNo % 1000000 == 0) Console.WriteLine("Line :  " + lineNo);
                yield return rec;
            }
        }

        public void WriteBinary(BinaryWriter writer)
        {
            writer.Write(Id);
            writer.Write(Label);
            for (var i = 0; i < FEATURE_COUNT; i++) writer.Write(Values[i]);
        }

        public RawRecord(BinaryReader reader)
        {
            Values = new int[FEATURE_COUNT];
            Id = reader.ReadInt32();
            Label = reader.ReadInt32();
            for (var i = 0; i < (FEATURE_COUNT); i++) Values[i] = reader.ReadInt32();
        }

        public static List<RawRecord> LoadCSV(string path)
        {
            var res = new List<RawRecord>();
            var lineNo = 0;
            foreach (var line in File.ReadLines(path))
            {
                lineNo++;
                if (lineNo == 1) continue;
                var rec = new RawRecord(line);

                if (lineNo % 1000000 == 0) Console.WriteLine("Line :  " + lineNo);
                res.Add(rec);
            }

            return res;
        }

        public static List<RawRecord> LoadBin(string path)
        {
            var fileStream = File.OpenRead(path);
            var deflateStream = new DeflateStream(fileStream, CompressionMode.Decompress);
            var reader = new BinaryReader(deflateStream);
            var res = new List<RawRecord>();
            var lineNo = 0;
            RawRecord rec = null;
            while (true)
            {
                lineNo++;
                try
                {
                    rec = new RawRecord(reader);
                }
                catch (EndOfStreamException ex)
                {
                    rec = null;
                }
                if (rec == null) break;
                res.Add(rec);
                if (lineNo % 1000000 == 0) Console.WriteLine("Line :  " + lineNo);
            }
            fileStream.Close();
            return res;
        }

        public static IEnumerable<RawRecord> EnumerateBinLines(string path)
        {
            var fileStream = File.OpenRead(path);
            var deflateStream = new DeflateStream(fileStream, CompressionMode.Decompress);
            var reader = new BinaryReader(deflateStream);
            var lineNo = 0;
            var quit = false;
            RawRecord rec;
            while (true)
            {
                lineNo++;
                try
                {
                    rec = new RawRecord(reader);
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

        public static void SaveBin(string path, List<RawRecord> records)
        {
            if (File.Exists(path)) File.Delete(path);
            var fileStream = File.OpenWrite(path);
            var deflateStream = new DeflateStream(fileStream, CompressionMode.Compress);
            var writer = new BinaryWriter(deflateStream);
            foreach (var rec in records)
            {
                rec.WriteBinary(writer);
            }
            writer.Flush();
            deflateStream.Flush();
            deflateStream.Close();
            fileStream.Close();
        }

        public static float[] GetMeans(params string[] srcPaths)
        {
            var means = new float[NUMERIC_COUNT];
            var totals = new double[NUMERIC_COUNT];
            var counts = new int[NUMERIC_COUNT];
            var label1count = 0;
            var recordCount = 0;
            foreach (var srcPath in srcPaths)
            {
                foreach (var src in RawRecord.EnumerateBinLines(srcPath))
                {
                    if (src.Label != 0) label1count++;
                    recordCount++;
                    for (var i = 0; i < NUMERIC_COUNT; i++)
                    {
                        var val = src.Values[i];
                        if (val > 0)
                        {
                            totals[i] += val;
                            counts[i] += 1;
                        }
                        else
                        {
                            recordCount = recordCount / 1;
                        }
                    }
                }
            }

            Console.WriteLine("1 labels : " + label1count + "//" + recordCount);
            for (var i = 0; i < NUMERIC_COUNT; i++) means[i] = (float)totals[i] / (float)counts[i];
            return means;
        }

        public static float[] GetStdDevs(float[] means, params string[] srcPaths)
        {
            var counts = new int[NUMERIC_COUNT];
            var squareds = new double[NUMERIC_COUNT];
            var res = new float[NUMERIC_COUNT];
            foreach (var srcPath in srcPaths)
            {
                foreach (var src in RawRecord.EnumerateBinLines(srcPath))
                {
                    for (var i = 0; i < NUMERIC_COUNT; i++)
                    {
                        var val = (double)src.Values[i];
                        if (val > 0)
                        {
                            var diff = val - (double)means[i];
                            var squared = diff * diff;
                            squareds[i] += squared;
                            counts[i] += 1;
                        }
                    }
                }
            }
            for (var i = 0; i < NUMERIC_COUNT; i++)
            {
                var squared = squareds[i];
                var count = counts[i];
                res[i] = (float)Math.Sqrt(squared / (double)count);
            }
            return res;
        }
    }
}

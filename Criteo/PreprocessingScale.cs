using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class PreprocessingScale
    {
        public static void ScaleNumericValues(string srcTrainPath, string srcTestPath, string dstTrainPath, string dstTestPath)
        {
            Console.WriteLine("Computing means");
            var means = OneHotRecord.GetMeans(srcTrainPath, srcTestPath);
            Console.WriteLine("Computing stddevs");
            var stdevs = OneHotRecord.GetStdDevs(means, srcTrainPath, srcTestPath);
            var paths = new List<string> { srcTrainPath + "^" + dstTrainPath, srcTestPath + "^" + dstTestPath };

            foreach (var pathItem in paths)
            {
                var pathItems = pathItem.Split('^');
                var srcPath = pathItems[0];
                var dstPath = pathItems[1];

                if (File.Exists(dstPath)) File.Delete(dstPath);
                var fileStream = File.OpenWrite(dstPath);
                var compressedStream = new DeflateStream(fileStream, CompressionMode.Compress);
                var writer = new BinaryWriter(compressedStream);

                Console.WriteLine("Standardizing" + Path.GetFileName(srcPath));
                var writeNo = 0;
                foreach (var rec in OneHotRecord.EnumerateBinLines(srcPath))
                {
                    for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++)
                    {
                        var val = rec.NumericData[i];
                        if (float.IsNaN(val))
                        {
                            rec.NumericData[i] = 0f;
                        }
                        else
                        {
                            var newVal = (rec.NumericData[i] - means[i]) / stdevs[i];
                            if (newVal > 3f) newVal = 3f;
                            if (newVal < -3f) newVal = -3f;
                            rec.NumericData[i] = newVal;
                        }
                    }
                    rec.WriteBinary(writer);
                    writeNo++;
                }
                writer.Flush();
                compressedStream.Flush();
                compressedStream.Close();
                fileStream.Close();
            }
        }
    }
}

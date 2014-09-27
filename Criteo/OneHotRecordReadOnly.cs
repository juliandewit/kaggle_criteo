using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class OneHotRecordReadOnly
    {
        public int Id;
        public int Label;
        public float[] NumericData;
        public int[] ReadOnlySparseDataIndices;
        public sbyte[] ReadOnlySparseValues;
        public byte ReadOnlySparseCurrentIndex;

        public OneHotRecordReadOnly()
        {
            NumericData = new float[RawRecord.NUMERIC_COUNT];
        }

        public OneHotRecordReadOnly(BinaryReader reader) : this()
        {
            Id = reader.ReadInt32();
            Label = reader.ReadInt32();
            var floatBytes = reader.ReadBytes(NumericData.Length * 4);
            Buffer.BlockCopy(floatBytes, 0, NumericData, 0, NumericData.Length * 4);
            var valCount = reader.ReadInt32();
            ReadOnlySparseDataIndices = new int[valCount];
            ReadOnlySparseValues = new sbyte[valCount];

            for (var i = 0; i < valCount; i++)
            {
                var key = reader.ReadInt32();
                var value = reader.ReadSByte();
                ReadOnlySparseDataIndices[ReadOnlySparseCurrentIndex] = key;
                ReadOnlySparseValues[ReadOnlySparseCurrentIndex] = value;
                ReadOnlySparseCurrentIndex++;
            }
        }

        public static IEnumerable<OneHotRecordReadOnly> EnumerateBinLines(string path)
        {
            var fileStream = File.OpenRead(path);
            var reader = new BinaryReader(fileStream);
            var lineNo = 0;
            OneHotRecordReadOnly rec;
            while (true)
            {
                lineNo++;
                try
                {
                    rec = new OneHotRecordReadOnly(reader);
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

        public static List<OneHotRecordReadOnly> LoadBinary(string path, bool decompress = true)
        {
            Console.WriteLine("Loading records into memory");
            var fileStream = File.OpenRead(path);
            var reader = new BinaryReader(fileStream);
            DeflateStream deflateStream = null;
            if (decompress)
            {
                deflateStream = new DeflateStream(fileStream, CompressionMode.Decompress);
                reader = new BinaryReader(deflateStream);
            }


            var res = new List<OneHotRecordReadOnly>();
            var lineNo = 0;
            while (true)
            {
                lineNo++;
                OneHotRecordReadOnly rec = null;
                try
                {

                    rec = new OneHotRecordReadOnly(reader);
                }
                catch (EndOfStreamException ex)
                {
                    rec = null;
                }

                if (rec == null) break;

                res.Add(rec);
                if (lineNo % 1000000 == 0) Console.WriteLine("Line :  " + lineNo);
            }

            if (decompress)
            {
                deflateStream.Close();
            }

            fileStream.Close();
            return res;
        }

        public void CopyDataToSparseArray(int[][] indices, float[][] values, int atIndex)
        {
            var len = RawRecord.NUMERIC_COUNT + this.ReadOnlySparseDataIndices.Length;
            indices[atIndex] = new int[len];
            values[atIndex] = new float[len];
            for (var i = 0; i < RawRecord.NUMERIC_COUNT; i++)
            {
                indices[atIndex][i] = i;
                values[atIndex][i] = this.NumericData[i];
            }

            for (var i = 0; i < this.ReadOnlySparseDataIndices.Length; i++)
            {
                var idx = i + RawRecord.NUMERIC_COUNT;
                var arrayIdx = this.ReadOnlySparseDataIndices[i];
                indices[atIndex][idx] = arrayIdx;
                var arrayValue = (float)this.ReadOnlySparseValues[i];
                values[atIndex][idx] = arrayValue; ;
            }

        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class RandomHelpers
    {
        private static Random _globalRandom = new Random();
        [ThreadStatic]
        private static Random _threadRandom;

        private static Random ThreadSafeRandom
        {
            get
            {
                if (_threadRandom == null)
                {
                    var seed = 0;
                    lock (_globalRandom)
                    {
                        seed = _globalRandom.Next();
                    }
                    _threadRandom = new Random(seed);
                }
                return _threadRandom;
            }
        }

        public static int Next(int maxValue)
        {
            return ThreadSafeRandom.Next(maxValue);
        }

        public static void NextBytes(byte[] buffer)
        {
            ThreadSafeRandom.NextBytes(buffer);
        }

        public static double NextDouble()
        {
            return ThreadSafeRandom.NextDouble();
        }

        public static float NextFloat()
        {
            return (float)ThreadSafeRandom.NextDouble();
        }

        public static float[] NextFloats(int count)
        {
            var res = new float[count];
            for (int i = 0; i < count; i++) res[i] = NextFloat();
            return res;
        }

        public static double GetRandomGaussian()
        {
            var u1 = ThreadSafeRandom.NextDouble(); //these are uniform(0,1) random doubles
            var u2 = ThreadSafeRandom.NextDouble();
            var res = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return res;
        }

        public static double GetRandomGaussian(double mean, double stdDev)
        {
            var res = GetRandomGaussian();
            res = mean + stdDev * res; //random normal(mean,stdDev^2)
            return res;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class Constants
    {
        // Value constants
        public const int VALUE_MISSING = Int32.MinValue;
        public const int VALUE_TRAINNOTTEST = 1;
        public const int VALUE_TESTNOTTRAIN = 2;
        public const int VALUE_LOWFREQUENCY = 3;


        public static Dictionary<int, int> FREQUENCY_FILTER_AGGRESSIVE = new Dictionary<int, int> { { 0, 0 } , { 1, 0 } , { 2, 10 } , { 3, 100 }, { 4, 50 }, { 5, 30} , { 6, 10} , { 7, 50 } ,
                                                        { 8, 50 } , { 9, 0 } , { 10, 50 }, { 11, 50 } , { 12, 100 } , { 13, 100 } , { 14, 0} ,
                                                        { 15, 50 } , { 16, 100 }, { 17, 0 } , {18, 100} , { 19, 100 } , { 20, 0 }, { 21, 100 } , { 22, 0 } ,
                                                        { 23, 0 } , { 24, 100} , { 25, 0 } , { 26, 100} };


        public static Dictionary<int, int> FREQUENCY_FILTER_MEDIUM = new Dictionary<int, int> { { 0, 0 } , { 1, 0 } , { 2, 0 } , { 3, 100 }, { 4, 50 }, { 5, 0} , { 6, 0} , { 7, 0 } ,
                                                          { 8, 0 } , { 9, 0 } , { 10, 10 }, { 11, 0 } , { 12, 100 } , { 13, 0 } , { 14, 0} ,
                                                          { 15, 25 } , { 16, 100 }, { 17, 0 } , {18, 0} , { 19, 0 } , { 20, 0 }, { 21, 100 } , { 22, 0 } ,
                                                          { 23, 0 } , { 24, 25} , { 25, 0 } , { 26, 25} };

        public static Dictionary<int, int> FREQUENCY_FILTER_MILD = new Dictionary<int, int> { { 0, 0 } , { 1, 0 } , { 2, 0 } , { 3, 50 }, { 4, 10 }, { 5, 0} , { 6, 0} , { 7, 0 } ,
                                                          { 8, 0 } , { 9, 0 } , { 10, 0 }, { 11, 0 } , { 12, 50 } , { 13, 0 } , { 14, 0} ,
                                                          { 15, 0 } , { 16, 50 }, { 17, 0 } , {18, 0} , { 19, 0 } , { 20, 0 }, { 21, 50 } , { 22, 0 } ,
                                                          { 23, 0 } , { 24, 10} , { 25, 0 } , { 26, 5}, { 27, 50 }, {28, 50 } };


        public static int[] ONEHOT_FEATURE_START_INDICES;
        public static int[] ONEHOT_FEATURE_VALUE_COUNTS;
        public static int HASH_SPACE_SIZE = 32768 * 2;
        public static int HASH_SPACE_START_INDEX;
        public static int NUMERICMISSING_COUNT;
        public static int NUMERICMISSING_START_INDEX;
        public static int TOTAL_VALUE_COUNT = 0;

        public static void InitOneHotIndices()
        {
            if (ONEHOT_FEATURE_START_INDICES == null)
            {
                ONEHOT_FEATURE_START_INDICES = new int[RawRecord.CATEGORICAL_COUNT + 1];
                ONEHOT_FEATURE_VALUE_COUNTS = new int[RawRecord.CATEGORICAL_COUNT + 1];

                // Number of different values for the small categories, they don't get hashed.
                ONEHOT_FEATURE_VALUE_COUNTS[1] = 1461;
                ONEHOT_FEATURE_VALUE_COUNTS[2] = 600;
                ONEHOT_FEATURE_VALUE_COUNTS[5] = 306;
                ONEHOT_FEATURE_VALUE_COUNTS[6] = 24;
                ONEHOT_FEATURE_VALUE_COUNTS[8] = 634;
                ONEHOT_FEATURE_VALUE_COUNTS[9] = 4;
                ONEHOT_FEATURE_VALUE_COUNTS[11] = 5750;
                ONEHOT_FEATURE_VALUE_COUNTS[13] = 3500;
                ONEHOT_FEATURE_VALUE_COUNTS[14] = 28;
                //CAT_COUNTS[15] = 14923;
                ONEHOT_FEATURE_VALUE_COUNTS[17] = 11;
                ONEHOT_FEATURE_VALUE_COUNTS[18] = 5800;
                ONEHOT_FEATURE_VALUE_COUNTS[19] = 2300;
                ONEHOT_FEATURE_VALUE_COUNTS[20] = 4;
                ONEHOT_FEATURE_VALUE_COUNTS[22] = 18;
                ONEHOT_FEATURE_VALUE_COUNTS[23] = 16;
                ONEHOT_FEATURE_VALUE_COUNTS[25] = 105;

                var index = RawRecord.NUMERIC_COUNT;
                for (int i = 0; i <= RawRecord.CATEGORICAL_COUNT; i++)
                {
                    ONEHOT_FEATURE_START_INDICES[i] = index;
                    index += ONEHOT_FEATURE_VALUE_COUNTS[i];
                }

                // Rest of categories get hashed 
                HASH_SPACE_START_INDEX = index;

                // We can also register when a numeric value is missing
                NUMERICMISSING_START_INDEX = HASH_SPACE_START_INDEX + HASH_SPACE_SIZE;
                NUMERICMISSING_COUNT = RawRecord.NUMERIC_COUNT;
                TOTAL_VALUE_COUNT = NUMERICMISSING_START_INDEX + NUMERICMISSING_COUNT;
                Console.WriteLine("Value count = " + TOTAL_VALUE_COUNT);
            }
        }

        public static int MINIBATCH_SIZE = 128; // minibatch size for the network
        public static int BATCHES_PER_SET = 100; // how many minibatches do WeakReference copy to the GPU in one go
    }
}

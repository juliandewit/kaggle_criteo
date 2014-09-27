using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Criteo
{
    public class SubmissionLine
    {
        public int Id;
        public float Chance;

        public string ToCsv()
        {
            return Id + "," + Chance.ToString(CultureInfo.InvariantCulture);
        }

        public static void SaveSubmission(string path, List<SubmissionLine> lines) 
        {
            Console.WriteLine("Writing submission : " + Path.GetFileName(path));

            var builder = new StringBuilder();
            builder.AppendLine("Id,Predicted");
            var sortLines = lines.OrderBy(x=>x.Id).ToList();
            if (sortLines.First().Id != 60000000) throw new Exception("First is should be 60000000");
            if (sortLines.Last().Id != 66042134) throw new Exception("Last is should be 66042134");
            if (sortLines.Count != 6042135) throw new Exception("# lines count should be 6042135");

            var changeGt05 = false;
            var changeGt10 = false;
            foreach (var line in sortLines) 
            {
                builder.AppendLine(line.ToCsv());
            }

            if (!changeGt05) throw new Exception("No chance gt 0.5");
            if (changeGt10) throw new Exception("A chance gt 1.0");

            File.WriteAllText(path, builder.ToString());
            Console.WriteLine("Submission written");
        }
    }
}

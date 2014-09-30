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

            foreach (var line in sortLines) 
            {
                builder.AppendLine(line.ToCsv());
            }
            
            File.WriteAllText(path, builder.ToString());
            Console.WriteLine("Submission written");
        }
    }
}

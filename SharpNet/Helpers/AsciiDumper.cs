using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNet
{
    public class AsciiDumper
    {
        Network _network;
        public AsciiDumper(Network net)
        {
            _network = net;
        }

        public void Export(string dir = null)
        {
            _network.CopyToHost();
            if (dir == null) dir = Directory.GetCurrentDirectory();
            _network.CopyToHost();
            if (!Directory.Exists(dir)) Directory.CreateDirectory(dir);
            var subDir = Path.Combine(dir, "asciidata");
            if (!Directory.Exists(subDir)) Directory.CreateDirectory(subDir);
            var files = Directory.GetFiles(subDir, "*.txt");
            foreach (var file in files) File.Delete(file);
            ReportLayers(dir, subDir);
        }

        public void ReportLayers(string dir, string subdir)
        {
            foreach (var layer in _network.Layers)
            {
                DumpLayer(layer, subdir);
            }
            DumpLayer(_network.LabelLayer, subdir);
        }

        public void DumpLayer(Layer layer, string dir)
        {
            layer.CopyToHost();
            var fileprefix = layer.Id + "_" + layer.GetType();
            DumpArray(layer, ArrayName.Outputs, fileprefix, dir);
            DumpArray(layer, ArrayName.Gradients, fileprefix, dir);
            DumpArray(layer, ArrayName.Weights, fileprefix, dir);
            DumpArray(layer, ArrayName.WeightUpdates, fileprefix, dir);
            DumpArray(layer, ArrayName.LastWeightUpdates, fileprefix, dir);
            DumpArray(layer, ArrayName.CorrectlyPredictedLabels, fileprefix, dir);
            DumpArray(layer, ArrayName.BiasWeights, fileprefix, dir);
            DumpArray(layer, ArrayName.BiasWeightUpdates, fileprefix, dir);
            DumpArray(layer, ArrayName.LastBiasWeightUpdates, fileprefix, dir);
        }

        public void DumpArray(Layer layer, ArrayName name, string filePrefix, string dir)
        {
            var arr = layer.GetArray(name);
            if (arr != null)
            {
                var txt = arr.GetTxt();
                var fileName = filePrefix + "_" + name.ToString() + ".txt";
                var path = Path.Combine(dir, fileName);
                File.WriteAllText(path, txt);
            }

        }
    }
}

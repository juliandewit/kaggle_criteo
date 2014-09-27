using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace SharpNet
{
    public enum SerializationOptions { SerializeWeights };
    public static class NetSerializer
    {
        public static void SerializeNet(string path, Network net)
        {
            var doc = new XDocument();
            var rootElement = new XElement("Network");
            var layersElement = new XElement("Layers");
            rootElement.Add(layersElement);
            foreach (var layer in net.Layers)
            {
                SerializeLayer(layersElement, layer);
            }
            doc.Add(rootElement);
            doc.Save(path);
        }

        public static void SerializeLayer(XElement parent, Layer layer)
        {
            var layerElement = new XElement("Layer");
            layerElement.AddElement("Id", layer.Id);
            layerElement.AddElement("Size", layer.Size.ToString());
            if (layer.HasWeights)
            {
                var weights = layer.GetArray(ArrayName.Weights);
                var biasWeights = layer.GetArray(ArrayName.BiasWeights);
                weights.CopyToHost();
                biasWeights.CopyToHost();
                layerElement.AddElement("Weights", SerializeArrayValues(weights));
                layerElement.AddElement("BiasWeights", SerializeArrayValues(biasWeights));
            }
            parent.Add(layerElement);
        }

        public static string SerializeArrayValues(CpuGpuArray array)
        {
            var res = String.Join("\n", array.CPUArray.Select(x => x.ToString(CultureInfo.InvariantCulture)));
            return res;
        }

        /// <summary>
        /// Load data into existing network
        /// </summary>
        public static void DeserializeNet(string path, Network net, params SerializationOptions[] options)
        {
            var doc = XDocument.Load(path);
            var rootElement = doc.Root;
            var layerElements = rootElement.Element("Layers").Elements("Layer");
            foreach (var layerElement in layerElements)
            {
                var id = layerElement.GetElementValue("Id");
                var layer = net.GetLayerById(id);
                if (layer != null)
                {
                    DeserializeLayer(layerElement, layer, options);
                }
            }
        }

        public static void DeserializeLayer(XElement layerElement, Layer layer, params SerializationOptions[] options)
        {
            if (layer.HasWeights)
            {
                var weightsElement = layerElement.Element("Weights");
                if (weightsElement == null) throw new Exception("Weights not found for layer : " + layer.Id);
                var weightArray = layer.GetArray(ArrayName.Weights);
                var newData = weightsElement.Value.Split('\n').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                if (newData.Length != weightArray.CPUArray.Length) throw new Exception("Size of weight arrays dont match");
                weightArray.CPUArray = newData;


                var biasWeightElement = layerElement.Element("BiasWeights");
                if (biasWeightElement == null) throw new Exception("BiasWeights not found for layer : " + layer.Id);
                var biasArray = layer.GetArray(ArrayName.BiasWeights);
                newData = biasWeightElement.Value.Split('\n').Select(x => float.Parse(x, CultureInfo.InvariantCulture)).ToArray();
                if (newData.Length != biasArray.CPUArray.Length) throw new Exception("Size of weight arrays dont match");
                biasArray.CPUArray = newData;
            }
        }

        public static XElement AddElement(this XElement pThis, string pName, string pValue)
        {
            var fElement = new XElement(pName);
            fElement.Value = pValue;
            pThis.Add(fElement);
            return fElement;
        }

        public static string GetElementValue(this XElement pThis, string pName)
        {
            var fElement = pThis.Element(pName);
            return fElement.Value;
        }



    }
}

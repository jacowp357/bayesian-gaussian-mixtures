using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler.Visualizers;
using Range = Microsoft.ML.Probabilistic.Models.Range;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace TestInfer
{
    class Program
    {
        static void Main(string[] args)
        {
            // number of data points
            int n = 100;

            // array that will host our data
            double[] data = new double[n];

            // reading the data from a comma separated .csv file
            String line = String.Empty;

            System.IO.StreamReader file = new System.IO.StreamReader("data.csv");

            int iter = 0;
            while ((line = file.ReadLine()) != null)
            {
                String[] parts_of_line = line.Split(',');
                for (int i = 0; i < parts_of_line.Length; i++)
                {
                    parts_of_line[i] = parts_of_line[i].Trim();
                    data[iter] = double.Parse(parts_of_line[i], System.Globalization.CultureInfo.InvariantCulture);
                }
                iter++;
            }

            // use "Ranges" and "VariableArrays" for efficiency see: https://dotnet.github.io/infer/userguide/Arrays%20and%20ranges.html 
            Range numDataPoints = new Range(n);

            // create a Gaussian distribution over a random variable called "mean"
            var mean = Variable.GaussianFromMeanAndPrecision(0, 1).Named("Mean");

            // create a variable array of doubles, which will host the observed data points
            VariableArray<double> x = Variable.Array<double>(numDataPoints).Named("x");

            // create a stream of Gaussian distributions using "ForEach" see: https://dotnet.github.io/infer/userguide/ForEach%20blocks.html
            x[numDataPoints] = Variable.GaussianFromMeanAndPrecision(mean, 1).ForEach(numDataPoints);

            // observe the data
            x.ObservedValue = data;

            // define the inference engine settings see: https://dotnet.github.io/infer/userguide/inference%20engine%20settings.html
            var engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            //engine.ShowFactorGraph = true;

            // infer the mean posterior Gaussian distribution 
            Gaussian postMean = engine.Infer<Gaussian>(mean);

            Console.WriteLine("Posterior Gaussian (Gaussian mean): {0}", postMean);

        }
    }
}

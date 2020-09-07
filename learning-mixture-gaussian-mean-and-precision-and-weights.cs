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

            Range numDataPoints = new Range(n);

            // number of Gaussian components
            Range k = new Range(3);

            // create a variable array of 3 Gaussian distributions depicting the mean of each Gaussian component
            VariableArray<double> means = Variable.Array<double>(k).Named("Means");
            means[k] = Variable.GaussianFromMeanAndPrecision(0, 1).ForEach(k);

            // create a variable array of 3 Gamma distributions depicting the precision of each Gaussian component
            VariableArray<double> precisions = Variable.Array<double>(k).Named("Precisions");
            precisions[k] = Variable.GammaFromShapeAndRate(2, 1).ForEach(k);

            // create a variable array of doubles, which will host the observed data points
            VariableArray<double> x = Variable.Array<double>(numDataPoints).Named("x");

            // create a variable array of ints, which is the discrete latent variable z 
            VariableArray<int> z = Variable.Array<int>(numDataPoints).Named("z");

            // create Dirichlet prior for mixture weights 
            Variable<Vector> weights = Variable.Dirichlet(k, new double[] { 1.1, 1.1, 1.1});

            // the using syntax see: https://dotnet.github.io/infer/userguide/ForEach%20blocks.html
            using (Variable.ForEach(numDataPoints))
            {
                // create uniform discrete distributions over the latent z variables
                z[numDataPoints] = Variable.Discrete(weights);

                // switch block allows variable-sized mixtures to be constructed see: https://dotnet.github.io/infer/userguide/Mixture%20of%20Gaussians%20tutorial.html
                using (Variable.Switch(z[numDataPoints]))
                {
                    x[numDataPoints] = Variable.GaussianFromMeanAndPrecision(means[z[numDataPoints]], precisions[z[numDataPoints]]);
                }
            }

            // break symmetry by random initialisation see: https://dotnet.github.io/infer/userguide/Mixture%20of%20Gaussians%20tutorial.html
            Discrete[] zinit = new Discrete[numDataPoints.SizeAsInt];
            for (int i = 0; i < zinit.Length; i++)
                zinit[i] = Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt);
            z.InitialiseTo(Distribution<int>.Array(zinit));

            // observe the data
            x.ObservedValue = data;

            var engine = new InferenceEngine();
            engine.Algorithm = new VariationalMessagePassing();
            //engine.ShowFactorGraph = true;

            // infer the mean posterior Gaussian distribution 
            Gaussian[] postMean = engine.Infer<Gaussian[]>(means);
            // infer the precision posterior Gamma distribution 
            Gamma[] postPrecision = engine.Infer<Gamma[]>(precisions);
            // infer the weights posterior discrete distribution 
            Discrete[] postZ = engine.Infer<Discrete[]>(z);
            // infer the mixture weights Dirichlet posterior  
            Dirichlet postWeights = engine.Infer<Dirichlet>(weights);

            for (int i = 0; i < 3; i++)
            {
                Console.WriteLine("Posterior Gaussian (Gaussian mean): {0}", postMean[i]);
                Console.WriteLine("Posterior Gamma (Gaussian precision): {0}", postPrecision[i]);
            }

            for (int i = 0; i < 5; i++)
            {
                Console.WriteLine("x = {0}: p(z) = {1}", data[i], postZ[i]);
            }

            Console.WriteLine("Posterior weight distribution: {0} ", postWeights.GetMean());
        }
    }
}

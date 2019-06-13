using System;
using System.Collections.Generic;

namespace NeralNetworks
{
    public class Neuron
    {
        public List<double> Weigths { get; }
        public List<double> Inputs { get; }
        public NeuronType NeuronType { get; }
        public double Output { get; private set; }
        public double Delta { get; private set; }


        public Neuron(int inputCount, NeuronType type = NeuronType.Normal)
        {
            NeuronType = type;
            Weigths = new List<double>();
            Inputs = new List<double>();
            InitWeigthRandomValues(inputCount);
        }


        private static readonly Random getrandom = new Random();

        public static int GetRandomNumber(int min = 0, int max = 1000)
        {
            lock (getrandom) // synchronize
            {
                return getrandom.Next(min, max);
            }
        }


        private void InitWeigthRandomValues(int inputCount)
        {

            for (int i = 0; i < inputCount; i++)
            {
                if (NeuronType == NeuronType.Input)
                {
                    Weigths.Add(1);
                }
                else
                {
                    double w = (double)GetRandomNumber() / 1000.0;
                    Weigths.Add(Math.Round(w, 2));
                }

                Inputs.Add(0);
            }
        }

        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            double sum = 0.0;

            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weigths[i];
            }

            if (NeuronType != NeuronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            
            return Output;
        }

        double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid / (1 - sigmoid);
            return result;
        }

        public void Learn(double error, double learnRate)
        {
            if (NeuronType == NeuronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weigths.Count; i++)
            {
                var weigth = Weigths[i];
                var input = Inputs[i];

                var newWeigth = weigth - input * Delta * learnRate;
                Weigths[i] = newWeigth;
            }             
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}

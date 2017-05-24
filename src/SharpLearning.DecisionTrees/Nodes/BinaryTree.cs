using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using System;
using System.Collections.Generic;

namespace SharpLearning.DecisionTrees.Nodes
{
    using System.Linq;

    using Microsoft.Win32.SafeHandles;

    using SharpLearning.Common.Interfaces;

    /// <summary>
    /// Binary tree 
    /// </summary>
    [Serializable]
    public abstract class BinaryTree : IPredictorModel<double>
    {
        /// <summary>
        /// Tree Nodes
        /// </summary>
        public readonly List<Node> Nodes;
        
        /// <summary>
        /// Leaf node probabilities
        /// </summary>
        public readonly List<double[]> Probabilities;

        /// <summary>
        /// Target names
        /// </summary>
        public readonly double[] TargetNames;

        /// <summary>
        /// Raw variable importance
        /// </summary>
        public readonly double[] VariableImportance;
        
        /// <summary>
        /// 
        /// </summary>
        /// <param name="nodes"></param>
        /// <param name="probabilities"></param>
        /// <param name="targetNames"></param>
        /// <param name="variableImportance"></param>
        public BinaryTree(List<Node> nodes, List<double[]> probabilities, double[] targetNames, 
            double[] variableImportance)
        {
            if (nodes == null) { throw new ArgumentNullException("nodes"); }
            if (probabilities == null) { throw new ArgumentNullException("probabilities"); }
            if (targetNames == null) { throw new ArgumentNullException("targetNames"); }
            if (variableImportance == null) { throw new ArgumentNullException("variableImportance"); }
            Nodes = nodes;
            Probabilities = probabilities;
            TargetNames = targetNames;
            VariableImportance = variableImportance;
        }

        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            return Predict(Nodes[0], observation);
        }

        /// <summary>
        /// Predicts a set of observations 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = this.Predict(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations, int[] indices)
        {
            var predictions = new double[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = this.Predict(observations.Row(indices[i]));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts using a continous node strategy
        /// </summary>
        /// <param name="root"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        protected double Predict(Node root, double[] observation)
        {
            return this.PredictNode(root, observation).Value;
        }

        /// <summary>
        /// Returns the prediction node using a continous node strategy
        /// </summary>
        /// <param name="root"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        protected Node PredictNode(Node root, double[] observation)
        {
            var currentNode = root;
            var currentIteration = 0;
            while (currentNode.FeatureIndex >= 0)
            {
                if (observation[currentNode.FeatureIndex] <= currentNode.Value)
                {
                    currentNode = this.Nodes[currentNode.LeftIndex];
                }
                else
                {
                    currentNode = this.Nodes[currentNode.RightIndex];
                }

                // make sure to prevent infinite loop
                // Tree might not be balanced, thus we cannot use something like Ceiling(Log(Nodes.Count, 2))
                // Maybe use Ceiling(Nodes.Count / 2.0) ? 
                if (currentIteration++ > this.Nodes.Count)
                {
                    throw new InvalidOperationException("The tree is degenerated.");
                }
            }

            return currentNode;
        }

        public double[] GetRawVariableImportance()
        {
            return this.VariableImportance;
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var max = this.VariableImportance.Max();

            var scaledVariableImportance = this.VariableImportance
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }
}

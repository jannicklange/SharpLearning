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
        /// <param name="node"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        protected double Predict(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                return node.Value;
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return Predict(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return Predict(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Returns the prediction node using a continous node strategy
        /// </summary>
        /// <param name="node"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        protected Node PredictNode(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                return node;
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return PredictNode(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return PredictNode(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
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

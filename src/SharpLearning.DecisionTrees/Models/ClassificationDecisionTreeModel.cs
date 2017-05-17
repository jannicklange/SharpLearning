using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.DecisionTrees.Models
{
    /// <summary>
    /// Classification Decision tree model
    /// </summary>
    [Serializable]
    public sealed class ClassificationDecisionTreeModel : BinaryTree, IPredictorModel<double>, IPredictorModel<ProbabilityPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="tree"></param>
        public ClassificationDecisionTreeModel(List<Node> nodes, List<double[]> probabilities, double[] targetNames,
            double[] variableImportance) : base(nodes, probabilities, targetNames, variableImportance)
        {
        }

        ProbabilityPrediction IPredictor<ProbabilityPrediction>.Predict(double[] observations)
        {
            return this.PredictProbability(observations);
        }

        ProbabilityPrediction[] IPredictor<ProbabilityPrediction>.Predict(F64Matrix observations)
        {
            return this.PredictProbability(observations);
        }

        ProbabilityPrediction[] IPredictor<ProbabilityPrediction>.Predict(F64Matrix observations, int[] indices)
        {
            return this.PredictProbability(observations, indices);
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public ProbabilityPrediction PredictProbability(double[] observation)
        {
            return PredictProbability(Nodes[0], observation);
        }

        /// <summary>
        /// Predicts a set of observations with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = this.PredictProbability(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Predicts the observation subset provided by indices with probabilities
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ProbabilityPrediction[] PredictProbability(F64Matrix observations, int[] indices)
        {
            var rows = observations.RowCount;
            var predictions = new ProbabilityPrediction[indices.Length];
            for (int i = 0; i < indices.Length; i++)
            {
                predictions[i] = this.PredictProbability(observations.Row(indices[i]));
            }

            return predictions;
        }

        /// <summary>
        /// Predict probabilities using a continous node strategy
        /// </summary>
        /// <param name="node"></param>
        /// <param name="observation"></param>
        /// <returns></returns>
        protected ProbabilityPrediction PredictProbability(Node node, double[] observation)
        {
            if (node.FeatureIndex == -1.0)
            {
                var probabilities = Probabilities[node.LeafProbabilityIndex];
                var targetProbabilities = new Dictionary<double, double>();

                for (int i = 0; i < TargetNames.Length; i++)
                {
                    targetProbabilities.Add(TargetNames[i], probabilities[i]);
                }

                return new ProbabilityPrediction(node.Value, targetProbabilities);
            }

            if (observation[node.FeatureIndex] <= node.Value)
            {
                return PredictProbability(Nodes[node.LeftIndex], observation);
            }
            else
            {
                return PredictProbability(Nodes[node.RightIndex], observation);
            }

            throw new InvalidOperationException("The tree is degenerated.");
        }

        /// <summary>
        /// Loads a ClassificationDecisionTreeModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static ClassificationDecisionTreeModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<ClassificationDecisionTreeModel>(reader);
        }

        /// <summary>
        /// Saves the ClassificationDecisionTreeModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}

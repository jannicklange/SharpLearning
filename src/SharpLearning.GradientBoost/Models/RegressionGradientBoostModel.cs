﻿using SharpLearning.Containers.Matrices;
using System;
using System.Linq;
using System.Collections.Generic;
using SharpLearning.Common.Interfaces;
using SharpLearning.InputOutput.Serialization;
using System.IO;
using SharpLearning.GradientBoost.GBMDecisionTree;

namespace SharpLearning.GradientBoost.Models
{
    /// <summary>
    /// 
    /// </summary>
    [Serializable]
    public sealed class RegressionGradientBoostModel : IPredictorModel<double>
    {
        /// <summary>
        /// 
        /// </summary>
        public readonly GBMTree[] Trees;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly double LearningRate;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly double InitialLoss;
        
        /// <summary>
        /// 
        /// </summary>
        public readonly int FeatureCount;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="trees"></param>
        /// <param name="learningRate"></param>
        /// <param name="initialLoss"></param>
        /// <param name="featureCount"></param>
        public RegressionGradientBoostModel(GBMTree[] trees, double learningRate, double initialLoss, int featureCount)
        {
            if (trees == null) { throw new ArgumentNullException("trees"); }
            Trees = trees;
            LearningRate = learningRate;
            InitialLoss = initialLoss;
            FeatureCount = featureCount;
        }

        /// <summary>
        /// Predicts a single observations using the combination of all predictors
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var prediction = InitialLoss;
            for (int i = 0; i < Trees.Length; i++)
            {
                prediction += LearningRate * Trees[i].Predict(observation);
            }

            return prediction;
        }

        /// <summary>
        /// Predicts a set of obervations using the combination of all predictors
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations)
        {
            var rows = observations.RowCount;
            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = Predict(observations.Row(i));
            }

            return predictions;
        }

        /// <summary>
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
            var m_rawVariableImportance = GetRawVariableImportance();
            var max = m_rawVariableImportance.Max();

            var scaledVariableImportance = m_rawVariableImportance
                .Select(v => (v / max) * 100.0)
                .ToArray();

            return featureNameToIndex.ToDictionary(kvp => kvp.Key, kvp => scaledVariableImportance[kvp.Value])
                        .OrderByDescending(kvp => kvp.Value)
                        .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        /// <summary>
        /// Gets the raw unsorted vatiable importance scores
        /// </summary>
        /// <returns></returns>
        public double[] GetRawVariableImportance()
        {
            var rawVariableImportance = new double[FeatureCount];
            foreach (var tree in Trees)
            {
                tree.AddRawVariableImportances(rawVariableImportance);
            }

            return rawVariableImportance;
        }

        /// <summary>
        /// Loads a GBMGradientBoostRegressorModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionGradientBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionGradientBoostModel>(reader);
        }

        /// <summary>
        /// Saves the GBMGradientBoostRegressorModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }

        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="indices">Subset of <paramref name="observations"/> to predict</param>
        /// <returns></returns>
        double[] IPredictor<double>.Predict(F64Matrix observations, int[] indices)
        {
            return Predict(observations, indices);
        }

        /// <summary>
        /// Predicts a set of observations
        /// </summary>
        /// <param name="observations">The observations</param>
        /// <param name="indices">Subset of <paramref name="observations"/> to predict</param>
        /// <returns></returns>
        public double[] Predict(F64Matrix observations, int[] indices)
        {
            var predictions = new double[indices.Length];
            var observation = new double[observations.ColumnCount];
            for (int i = 0; i < indices.Length; i++)
            {
                observations.Row(indices[i], observation);
                predictions[i] = Predict(observation);
            }

            return predictions;
        }
    }
}

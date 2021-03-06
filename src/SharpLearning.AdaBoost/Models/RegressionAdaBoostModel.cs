﻿using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Common.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using SharpLearning.InputOutput.Serialization;

namespace SharpLearning.AdaBoost.Models
{
    /// <summary>
    /// AdaBoost regression model. Consist of a series of tree model and corresponding weights
    /// </summary>
    [Serializable]
    public sealed class RegressionAdaBoostModel : IPredictorModel<double>
    {
        readonly double[] m_modelWeights;
        readonly RegressionDecisionTreeModel[] m_models;
        readonly double[] m_rawVariableImportance;
        readonly double[] m_predictions;

        /// <summary>
        /// AdaBoost regression model. Consist of a series of tree model and corresponding weights
        /// </summary>
        /// <param name="models"></param>
        /// <param name="modelWeights"></param>
        /// <param name="rawVariableImportance"></param>
        public RegressionAdaBoostModel(RegressionDecisionTreeModel[] models, double[] modelWeights,
            double[] rawVariableImportance)
        {
            if (models == null) { throw new ArgumentNullException("models"); }
            if (modelWeights == null) { throw new ArgumentNullException("modelWeights"); }
            if (rawVariableImportance == null) { throw new ArgumentNullException("rawVariableImportance"); }

            m_models = models;
            m_modelWeights = modelWeights;
            m_rawVariableImportance = rawVariableImportance;
            m_predictions = new double[m_models.Length];
        }

        /// <summary>
        /// Predicts a single observations using weighted median
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public double Predict(double[] observation)
        {
            var count = m_models.Length;

            for (int i = 0; i < count; i++)
            {
                m_predictions[i] = m_models[i].Predict(observation);
            }

            var weights = m_modelWeights.ToArray();
            m_predictions.SortWith(weights);

            var prediction = m_predictions.WeightedMedian(weights);

            return prediction;
        }

        /// <summary>
        /// Predicts a set of obervations using weighted median
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
        /// Returns the rescaled (0-100) and sorted variable importance scores with corresponding name
        /// </summary>
        /// <param name="featureNameToIndex"></param>
        /// <returns></returns>
        public Dictionary<string, double> GetVariableImportance(Dictionary<string, int> featureNameToIndex)
        {
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
            return m_rawVariableImportance;
        }

        /// <summary>
        /// Loads a RegressionAdaBoostModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionAdaBoostModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionAdaBoostModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionAdaBoostModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize(this, writer);
        }
    }
}

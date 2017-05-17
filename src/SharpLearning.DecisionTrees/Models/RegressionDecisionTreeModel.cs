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
    using SharpLearning.Containers;

    /// <summary>
    /// Regression Decision tree model
    /// </summary>
    [Serializable]
    public sealed class RegressionDecisionTreeModel : BinaryTree, IPredictorModel<double>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="tree"></param>
        public RegressionDecisionTreeModel(List<Node> nodes, List<double[]> probabilities, double[] targetNames,
            double[] variableImportance) : base(nodes, probabilities, targetNames, variableImportance)
        {
        }

        /// <summary>
        /// Loads a RegressionDecisionTreeModel.
        /// </summary>
        /// <param name="reader"></param>
        /// <returns></returns>
        public static RegressionDecisionTreeModel Load(Func<TextReader> reader)
        {
            return new GenericXmlDataContractSerializer()
                .Deserialize<RegressionDecisionTreeModel>(reader);
        }

        /// <summary>
        /// Saves the RegressionDecisionTreeModel.
        /// </summary>
        /// <param name="writer"></param>
        public void Save(Func<TextWriter> writer)
        {
            new GenericXmlDataContractSerializer()
                .Serialize<RegressionDecisionTreeModel>(this, writer);
        }
    }
}

using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using SharpLearning.Common.Interfaces;

namespace SharpLearning.DecisionTrees.Learners
{
    using System;

    /// <summary>
    /// Trains a Regression Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public sealed class RegressionDecisionTreeLearner : DecisionTreeLearner<RegressionDecisionTreeModel>,
        IIndexedLearner<double>, ILearner<double>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="minimumSplitSize">The minimum size </param>
        /// <param name="featuresPrSplit">The number of features to be selected between at each split</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="seed">Seed for feature selection if number of features pr split is not equal 
        /// to the total amount of features in observations. The features will be selected at random for each split</param>
        public RegressionDecisionTreeLearner(int maximumTreeDepth=2000, int minimumSplitSize=1, int featuresPrSplit=0, double minimumInformationGain=0.000001, int seed=42)
            : base(new DepthFirstTreeBuilder<RegressionDecisionTreeModel>(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, new OnlyUniqueThresholdsSplitSearcher(minimumSplitSize), 
                   new RegressionImpurityCalculator()))
        {
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Private explicit interface implementation for learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }
    }
}

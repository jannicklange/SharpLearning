using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.Learners
{
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.Models;
    using SharpLearning.DecisionTrees.SplitSearchers;
    using SharpLearning.DecisionTrees.TreeBuilders;

    public sealed class RegressionDecisionTreeLearner : GenericRegressionDecisionTreeLearner<DepthFirstTreeBuilder<RegressionDecisionTreeModel, OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>, RegressionDecisionTreeModel, OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>
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
        public RegressionDecisionTreeLearner(int maximumTreeDepth = 2000, int featuresPrSplit = 0, double minimumInformationGain = 0.000001, int seed = 42, int minimumSplitSize = 1)
            : base(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, minimumSplitSize)
        {
        }

        public RegressionDecisionTreeLearner(
            int maximumTreeDepth,
            int featuresPrSplit,
            double minimumInformationGain,
            int seed,
            OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator> searcher,
            RegressionImpurityCalculator calculator) :
            base(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, searcher, calculator)
        {
        }

        public RegressionDecisionTreeLearner(DepthFirstTreeBuilder<RegressionDecisionTreeModel, OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator> builder) : base(builder)
        {
        }
    }
}

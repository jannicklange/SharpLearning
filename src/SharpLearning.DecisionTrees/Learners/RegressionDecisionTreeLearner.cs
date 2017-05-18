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
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.TreeBuilders
{
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.Learners;
    using SharpLearning.DecisionTrees.Models;
    using SharpLearning.DecisionTrees.SplitSearchers;

    /// <summary>
    /// Concrete implementation of the <see cref="DepthFirstTreeBuilder{TTreeType,TSplitSearcher,TImpurityCalculator}"/> so that one does not need to fiddle around with all the generic parameter references when creating a <see cref="ClassificationDecisionTreeLearner"/>.
    /// </summary>
    public sealed class ClassificationDepthFirstTreeBuilder : DepthFirstTreeBuilder<ClassificationDecisionTreeModel, OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>
    {
        public ClassificationDepthFirstTreeBuilder(int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, int seed, OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator> splitSearcher, GiniClasificationImpurityCalculator impurityCalculator)
            : base(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, splitSearcher, impurityCalculator)
        {
        }
    }
}

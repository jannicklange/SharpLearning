using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using System;

namespace SharpLearning.DecisionTrees.Test.TreeBuilders
{
    using SharpLearning.DecisionTrees.Models;

    [TestClass]
    public class DepthFirstTreeBuilderTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidMaximumTreeSize()
        {
            new DepthFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(0, 1, 0.1, 42, new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1), new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidFeaturesPrSplit()
        {
            new DepthFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(1, -1, 0.1, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void DepthFirstTreeBuilder_InvalidMinimumInformationGain()
        {
            new DepthFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(1, 1, 0, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }
    }
}

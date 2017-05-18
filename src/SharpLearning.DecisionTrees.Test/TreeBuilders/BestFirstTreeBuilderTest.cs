﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.DecisionTrees.TreeBuilders;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System;
using System.IO;

namespace SharpLearning.DecisionTrees.Test.TreeBuilders
{
    [TestClass]
    public class BestFirstTreeBuilderTest
    {
        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BestFirstTreeBuilder_InvalidMaximumTreeSize()
        {
            new BestFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(0, 2, 1, 0.1, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BestFirstTreeBuilder_InvalidMaximumLeafCount()
        {
            new BestFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(1, 1, 1, 0.1, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }


        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BestFirstTreeBuilder_InvalidFeaturesPrSplit()
        {
            new BestFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(1, 2, -1, 0.1, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void BestFirstTreeBuilder_InvalidMinimumInformationGain()
        {
            new BestFirstTreeBuilder<ClassificationDecisionTreeModel, LinearSplitSearcher<GiniClasificationImpurityCalculator>, GiniClasificationImpurityCalculator>(1, 2, 1, 0, 42,
                new LinearSplitSearcher<GiniClasificationImpurityCalculator>(1),
                new GiniClasificationImpurityCalculator());
        }

        [TestMethod]
        public void BestFirstTreeBuilder_Build_Full_Tree()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut =
                new DecisionTreeLearner<
                    BestFirstTreeBuilder<ClassificationDecisionTreeModel,
                        OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                        GiniClasificationImpurityCalculator>, ClassificationDecisionTreeModel,
                    OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                    GiniClasificationImpurityCalculator>(
                    new BestFirstTreeBuilder<ClassificationDecisionTreeModel,
                        OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                        GiniClasificationImpurityCalculator>(
                        2000,
                        2000,
                        observations.ColumnCount,
                        0.000001,
                        42,
                        new OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>(1),
                        new GiniClasificationImpurityCalculator()));

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.0, actual, 0.00001);
        }


        [TestMethod]
        public void BestFirstTreeBuilder_Build_Leaf_Nodes_4()
        {
            var parser = new CsvParser(() => new StringReader(Resources.Glass));
            var observations = parser.EnumerateRows(v => v != "Target").ToF64Matrix();
            var targets = parser.EnumerateRows("Target").ToF64Vector();
            var rows = targets.Length;

            var sut =
                new DecisionTreeLearner<
                    BestFirstTreeBuilder<ClassificationDecisionTreeModel,
                        OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                        GiniClasificationImpurityCalculator>, ClassificationDecisionTreeModel,
                    OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                    GiniClasificationImpurityCalculator>(
                    new BestFirstTreeBuilder<ClassificationDecisionTreeModel,
                        OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>,
                        GiniClasificationImpurityCalculator>(
                        2000,
                        4,
                        observations.ColumnCount,
                        0.000001,
                        42,
                        new OnlyUniqueThresholdsSplitSearcher<GiniClasificationImpurityCalculator>(1),
                        new GiniClasificationImpurityCalculator()));

            var model = sut.Learn(observations, targets);

            var predictions = model.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.37383177570093457, actual, 0.00001);
        }
    }
}

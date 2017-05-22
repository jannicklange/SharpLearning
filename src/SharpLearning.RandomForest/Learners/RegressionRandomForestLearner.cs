using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.Common.Interfaces;
using SharpLearning.RandomForest.Models;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SharpLearning.RandomForest.Learners
{
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.Nodes;
    using SharpLearning.DecisionTrees.SplitSearchers;
    using SharpLearning.DecisionTrees.TreeBuilders;

    /// <summary>
    /// Trains a regression random forest
    /// http://en.wikipedia.org/wiki/Random_forest
    /// http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm
    /// </summary>
    public sealed class RegressionRandomForestLearner<TSplitSearcher, TImpurityCalculator> : GenericRandomizedForestBase<RegressionForestModel, RegressionDecisionTreeModel>, IIndexedLearner<double>, ILearner<double> 
                                                        where TSplitSearcher : ISplitSearcher<TImpurityCalculator>
                                                        where TImpurityCalculator : IImpurityCalculator
    {
        /// <summary>
        /// The random forest is an ensemble learner consisting of a series of randomized decision trees
        /// </summary>
        /// <param name="trees">Number of trees to use in the ensemble</param>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="maximumTreeDepth">The maximal tree depth before a leaf is generated</param>
        /// <param name="featuresPrSplit">Number of features used at each split in each tree</param>
        /// <param name="minimumInformationGain">The minimum improvement in information gain before a split is made</param>
        /// <param name="subSampleRatio">The ratio of observations sampled with replacement for each tree. 
        /// Default is 1.0 sampling the same count as the number of observations in the input. 
        /// If below 1.0 the algorithm changes to random patches</param>
        /// <param name="seed">Seed for the random number generator</param>
        /// <param name="runParallel">Use multi threading to speed up execution (default is true)</param>
        public RegressionRandomForestLearner(int trees = 100, int minimumSplitSize = 1, int maximumTreeDepth = 2000,
            int featuresPrSplit = 0, double minimumInformationGain = .000001, double subSampleRatio = 1.0, int seed = 42, bool runParallel = true)
            : base(trees, minimumSplitSize, maximumTreeDepth, featuresPrSplit, minimumInformationGain, subSampleRatio, seed, runParallel)
        {
        }

        protected override RegressionDecisionTreeModel CallCreateTree(F64Matrix observations, double[] targets, int[] indices)
        {
            Random rng;
            lock (this.rngLock)
            {
                var seed = this.m_random.Next();
                rng = new Random(seed);
            }

            var model = base.CreateTree<
                        GenericRegressionDecisionTreeLearner<
                            DepthFirstTreeBuilder<
                                RegressionDecisionTreeModel, 
                                TSplitSearcher, 
                                TImpurityCalculator>, 
                            RegressionDecisionTreeModel,
                            TSplitSearcher, 
                            TImpurityCalculator>,
                        DepthFirstTreeBuilder<
                            RegressionDecisionTreeModel, 
                            TSplitSearcher, 
                            TImpurityCalculator>, 
                        TSplitSearcher, 
                        TImpurityCalculator>(
                            observations,
                            targets,
                            indices,
                            rng,
                            // Parameters for GenericRegressionDecisionTreeLearner
                            this.m_maximumTreeDepth,
                            this.m_featuresPrSplit,
                            this.m_minimumInformationGain,
                            rng.Next(),
                            this.m_minimumSplitSize);

            return model;
        }
    }
}

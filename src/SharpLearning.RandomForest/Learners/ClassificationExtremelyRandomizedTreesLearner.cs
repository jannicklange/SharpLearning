using SharpLearning.Containers;
using SharpLearning.Containers.Matrices;
using SharpLearning.DecisionTrees.ImpurityCalculators;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.SplitSearchers;
using SharpLearning.DecisionTrees.TreeBuilders;
using SharpLearning.Common.Interfaces;
using SharpLearning.RandomForest.Models;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace SharpLearning.RandomForest.Learners
{
    /// <summary>
    /// Learns a classification version of Extremely randomized trees
    /// http://www.montefiore.ulg.ac.be/~ernst/uploads/news/id63/extremely-randomized-trees.pdf
    /// </summary>
    public sealed class ClassificationExtremelyRandomizedTreesLearner : GenericRandomizedForestBase<ClassificationForestModel, ClassificationDecisionTreeModel>, IIndexedLearner<double>, IIndexedLearner<ProbabilityPrediction>,
        ILearner<double>, ILearner<ProbabilityPrediction>
    {
        /// <summary>
        /// The extremely randomized trees learner is an ensemble learner consisting of a series of randomized decision trees. 
        /// It takes the randomization a step futher than random forest and also select the splits randomly
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
        public ClassificationExtremelyRandomizedTreesLearner(int trees = 100, int minimumSplitSize = 1, int maximumTreeDepth = 2000, 
            int featuresPrSplit = 0, double minimumInformationGain = .000001, double subSampleRatio = 1.0, int seed = 42, bool runParallel = true)
            : base(trees, minimumSplitSize, maximumTreeDepth, featuresPrSplit, minimumInformationGain, subSampleRatio, seed, runParallel)
        {
        }

        /// <summary>
        /// Learns a classification Extremely randomized trees model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public ClassificationForestModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Learns a classification Extremely randomized trees model
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public ClassificationForestModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            if (m_featuresPrSplit == 0)
            {
                var count = (int)Math.Sqrt(observations.ColumnCount);
                m_featuresPrSplit = count <= 0 ? 1 : count;
            }

            var results = new ConcurrentBag<ClassificationDecisionTreeModel>();

            if (!m_runParallel)
            {
                for (int i = 0; i < m_trees; i++)
                {
                    results.Add(this.CallCreateTree(observations, targets, indices));
                };
            }
            else
            {
                var workItems = Enumerable.Range(0, m_trees).ToArray();
                var rangePartitioner = Partitioner.Create(workItems, true);
                Parallel.ForEach(rangePartitioner, (work, loopState) =>
                {
                    results.Add(this.CallCreateTree(observations, targets, indices));
                });
            }

            var models = results.ToArray();
            var rawVariableImportance = VariableImportance(models, observations.ColumnCount);

            return new ClassificationForestModel(models, rawVariableImportance);
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
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> IIndexedLearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        /// <summary>
        /// Private explicit interface implementation for indexed probability learning.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        IPredictorModel<ProbabilityPrediction> ILearner<ProbabilityPrediction>.Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets);
        }

        double[] VariableImportance(ClassificationDecisionTreeModel[] models, int numberOfFeatures)
        {
            var rawVariableImportance = new double[numberOfFeatures];

            foreach (var model in models)
            {
                var modelVariableImportance = model.GetRawVariableImportance();

                for (int j = 0; j < modelVariableImportance.Length; j++)
                {
                    rawVariableImportance[j] += modelVariableImportance[j];
                }
            }
            return rawVariableImportance;
        }
        protected override ClassificationDecisionTreeModel CallCreateTree(F64Matrix observations, double[] targets, int[] indices)
        {
            int seed1 = 0, seed2 = 0;
            Random rng;
            lock (this.rngLock)
            {
                // explcicitly query m_random in this order so that the old unit tests get their expected seeds
                var rngSeed = this.m_random.Next();
                rng = new Random(rngSeed);
                seed1 = this.m_random.Next();
                seed2 = this.m_random.Next();
                
            }

            var model = base.CreateTree<GenericRegressionDecisionTreeLearner<
                            DepthFirstTreeBuilder<
                                ClassificationDecisionTreeModel,
                                RandomSplitSearcher<GiniClasificationImpurityCalculator>,
                                GiniClasificationImpurityCalculator>,
                            ClassificationDecisionTreeModel,
                            RandomSplitSearcher<GiniClasificationImpurityCalculator>,
                            GiniClasificationImpurityCalculator>,
                    DepthFirstTreeBuilder<
                            ClassificationDecisionTreeModel,
                            RandomSplitSearcher<GiniClasificationImpurityCalculator>,
                            GiniClasificationImpurityCalculator>,
                    RandomSplitSearcher<GiniClasificationImpurityCalculator>,
                    GiniClasificationImpurityCalculator>(observations, targets, indices, rng,
                    // params for GenericRegressionDecisionTreeLearner
                    this.m_maximumTreeDepth,
                    this.m_featuresPrSplit,
                    this.m_minimumInformationGain,
                    seed1,
                    this.m_minimumSplitSize,
                    seed2);

            return model;
        }
    }
}

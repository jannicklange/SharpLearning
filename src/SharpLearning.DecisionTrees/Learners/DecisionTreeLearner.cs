using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.DecisionTrees.TreeBuilders;
using System;
using System.Linq;

namespace SharpLearning.DecisionTrees.Learners
{
    using SharpLearning.Common.Interfaces;
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.SplitSearchers;

    /// <summary>
    /// Learns a Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public unsafe class DecisionTreeLearner<TTreeBuilder, TTreeType, TSplitSearcher, TImpurityCalculator> : IIndexedLearner<double>, ILearner<double> where TTreeType : BinaryTree
                                                                                where TSplitSearcher : ISplitSearcher<TImpurityCalculator>
                                                                                where TImpurityCalculator : IImpurityCalculator
                                                                                where TTreeBuilder : ITreeBuilder<TTreeType, TSplitSearcher, TImpurityCalculator>
    {
        readonly TTreeBuilder m_treeBuilder;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="treeBuilder"></param>
        public DecisionTreeLearner(TTreeBuilder treeBuilder)
        {
            if (treeBuilder == null) { throw new ArgumentNullException("treeBuilder"); }
            m_treeBuilder = treeBuilder;
        }

        public DecisionTreeLearner(int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, int seed, TSplitSearcher searcher, TImpurityCalculator calculator)
        {
            var treeBuilder =
                (TTreeBuilder)
                    Activator.CreateInstance(typeof(TTreeBuilder), maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, searcher, calculator);
            this.m_treeBuilder = treeBuilder;
        }

        public DecisionTreeLearner(int maximumTreeDepth, int featuresPrSplit, double minimumInformationGain, int seed, params object[] splitSearcherCtorArgs) :
            this(maximumTreeDepth, featuresPrSplit, minimumInformationGain, seed, (TSplitSearcher)Activator.CreateInstance(typeof(TSplitSearcher), splitSearcherCtorArgs), (TImpurityCalculator)Activator.CreateInstance(typeof(TImpurityCalculator)))
        {
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <returns></returns>
        public TTreeType Learn(F64Matrix observations, double[] targets)
        {
            return Learn(observations, targets, new double[0]);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets.
        /// Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        public TTreeType Learn(F64Matrix observations, double[] targets, double[] weights)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return Learn(observations, targets, indices, weights);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public TTreeType Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices, new double[0]);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times. Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights">Provide weights inorder to weigh each sample separetely</param>
        /// <returns></returns>
        public TTreeType Learn(F64Matrix observations, double[] targets, int[] indices, double[] weights)
        {
            using (var pinnedFeatures = observations.GetPinnedPointer())
            {
                return Learn(pinnedFeatures.View(), targets, indices, weights);
            }
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times.
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <returns></returns>
        public TTreeType Learn(F64MatrixView observations, double[] targets, int[] indices)
        {
            return Learn(observations, targets, indices, new double[0]);
        }

        /// <summary>
        /// Learns a decision tree from the provided observations and targets but limited to the observation indices provided by indices.
        /// Indices can contain the same index multiple times. Weights can be provided in order to weight each sample individually
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights">Provide weights inorder to weigh each sample separetely</param>
        /// <returns></returns>
        public TTreeType Learn(F64MatrixView observations, double[] targets, int[] indices, double[] weights)
        {
            return m_treeBuilder.Build(observations, targets, indices, weights);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return this.Learn(observations, targets, indices);
        }

        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return this.Learn(observations, targets);
        }
    }
}

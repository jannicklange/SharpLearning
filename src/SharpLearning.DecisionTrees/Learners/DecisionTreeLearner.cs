﻿using SharpLearning.Containers.Matrices;
using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;
using SharpLearning.DecisionTrees.TreeBuilders;
using System;
using System.Linq;

namespace SharpLearning.DecisionTrees.Learners
{
    /// <summary>
    /// Learns a Decision tree
    /// http://en.wikipedia.org/wiki/Decision_tree_learning
    /// </summary>
    public abstract unsafe class DecisionTreeLearner<TTreeType> where TTreeType : BinaryTree
    {
        readonly ITreeBuilder<TTreeType> m_treeBuilder;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="treeBuilder"></param>
        public DecisionTreeLearner(ITreeBuilder<TTreeType> treeBuilder)
        {
            if (treeBuilder == null) { throw new ArgumentNullException("treeBuilder"); }
            m_treeBuilder = treeBuilder;
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
    }
}

﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.RandomForest.Learners
{
    using System.Collections.Concurrent;

    using SharpLearning.Common.Interfaces;
    using SharpLearning.Containers;
    using SharpLearning.Containers.Matrices;
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.Learners;
    using SharpLearning.DecisionTrees.Nodes;
    using SharpLearning.DecisionTrees.SplitSearchers;
    using SharpLearning.DecisionTrees.TreeBuilders;
    using SharpLearning.RandomForest.Models;

    public abstract class GenericRandomizedForestBase<TForestModel, TTreeTypeModel> : IIndexedLearner<double>,
        ILearner<double>
        where TTreeTypeModel : BinaryTree
        where TForestModel : IPredictorModel<double>
    {
        protected readonly int m_trees;
        protected int m_featuresPrSplit;
        protected readonly int m_minimumSplitSize;
        protected readonly double m_subSampleRatio;
        protected readonly double m_minimumInformationGain;
        protected readonly int m_maximumTreeDepth;
        protected readonly Random m_random;
        protected readonly bool m_runParallel;

        protected object rngLock = new object();

        public GenericRandomizedForestBase(int trees = 100, int minimumSplitSize = 1, int maximumTreeDepth = 2000,
            int featuresPrSplit = 0, double minimumInformationGain = .000001, double subSampleRatio = 1.0, int seed = 42, bool runParallel = true)
        {
            if (trees < 1) { throw new ArgumentException("trees must be at least 1"); }
            if (featuresPrSplit < 0) { throw new ArgumentException("features pr split must be at least 1"); }
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            if (maximumTreeDepth <= 0) { throw new ArgumentException("maximum tree depth must be larger than 0"); }
            if (minimumInformationGain <= 0) { throw new ArgumentException("minimum information gain must be larger than 0"); }

            m_trees = trees;
            m_minimumSplitSize = minimumSplitSize;
            m_maximumTreeDepth = maximumTreeDepth;
            m_featuresPrSplit = featuresPrSplit;
            m_minimumInformationGain = minimumInformationGain;
            m_subSampleRatio = subSampleRatio;
            m_runParallel = runParallel;

            m_random = new Random(seed);
        }

        IPredictorModel<double> IIndexedLearner<double>.Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            return this.Learn(observations, targets, indices);
        }

        public virtual TForestModel Learn(F64Matrix observations, double[] targets)
        {
            var indices = Enumerable.Range(0, targets.Length).ToArray();
            return this.Learn(observations, targets, indices);
        }

        IPredictorModel<double> ILearner<double>.Learn(F64Matrix observations, double[] targets)
        {
            return this.Learn(observations, targets);
        }

        public virtual TForestModel Learn(F64Matrix observations, double[] targets, int[] indices)
        {
            if (m_featuresPrSplit == 0)
            {
                var count = (int)(observations.ColumnCount / 3.0);
                m_featuresPrSplit = count <= 0 ? 1 : count;
            }

            var results = new ConcurrentBag<TTreeTypeModel>();

            if (!m_runParallel)
            {
                for (int i = 0; i < m_trees; i++)
                {
                    results.Add(CallCreateTree(observations, targets, indices));
                };
            }
            else
            {
                var workItems = Enumerable.Range(0, m_trees).ToArray();
                var rangePartitioner = Partitioner.Create(workItems, true);
                Parallel.ForEach(rangePartitioner, (work, loopState) =>
                {
                    results.Add(CallCreateTree(observations, targets, indices));
                });
            }

            var models = results.ToArray();
            var rawVariableImportance = VariableImportance(models, observations.ColumnCount);

            return (TForestModel)Activator.CreateInstance(typeof(TForestModel), models, rawVariableImportance);
        }

        protected virtual TTreeTypeModel CreateTree<TTreeLearner, TTreeBuilder, TSplitSearcher, TImpurityCalculator>(F64Matrix observations, double[] targets, int[] indices, Random random, params object[] tTreeLearnerParams)
            where TTreeLearner : DecisionTreeLearner<TTreeBuilder, TTreeTypeModel, TSplitSearcher, TImpurityCalculator> 
            where TImpurityCalculator : IImpurityCalculator 
            where TSplitSearcher : ISplitSearcher<TImpurityCalculator>
            where TTreeBuilder : ITreeBuilder<TTreeTypeModel, TSplitSearcher, TImpurityCalculator>
        {
            var learner = (TTreeLearner)Activator.CreateInstance(typeof(TTreeLearner), tTreeLearnerParams);

            //var learner = new DecisionTreeLearner(
            //    new DepthFirstTreeBuilder(m_maximumTreeDepth,
            //        m_featuresPrSplit,
            //        m_minimumInformationGain,
            //        m_random.Next(),
            //        new RandomSplitSearcher(m_minimumSplitSize, m_random.Next()),
            //        new RegressionImpurityCalculator()));

            var treeIndicesLength = (int)Math.Round(m_subSampleRatio * (double)indices.Length);
            var treeIndices = new int[treeIndicesLength];

            for (int j = 0; j < treeIndicesLength; j++)
            {
                treeIndices[j] = indices[random.Next(indices.Length)];
            }

            var model = learner.Learn(observations, targets, treeIndices);
            return model;
        }

        protected abstract TTreeTypeModel CallCreateTree(F64Matrix observations, double[] targets, int[] indices);

        protected virtual double[] VariableImportance(TTreeTypeModel[] models, int numberOfFeatures)
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
    }
}

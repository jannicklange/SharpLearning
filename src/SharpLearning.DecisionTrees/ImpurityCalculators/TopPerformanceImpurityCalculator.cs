using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    using SharpLearning.Containers.Views;

    /// <summary>
    /// Computes a custom split criterion. Goal is to separate the top performing q% from the remaining nodes.
    /// Should be used together with <see cref="TopPerformanceSplitSearcher"/>.
    /// </summary>
    public class TopPerformanceImpurityCalculator : ITopImpurityCalculator
    {
        Interval1D m_interval;
        int m_currentPosition;

        double m_weightedTotal = 0.0;
        double m_weightedLeft = 0.0;
        double m_weightedRight = 0.0;

        double m_meanLeft = 0.0;
        double m_meanRight = 0.0;
        double m_meanTotal = 0.0;

        double m_sqSumLeft = 0.0;
        double m_sqSumRight = 0.0;
        double m_sqSumTotal = 0.0;

        double m_varLeft = 0.0;
        double m_varRight = 0.0;

        double m_sumLeft = 0.0;
        double m_sumRight = 0.0;
        double m_sumTotal = 0.0;

        double[] m_targets;
        double[] m_weights;

        protected double TopThresoldValue;

        protected bool[] FeatureSortedTopPerformers;

        /// <summary>
        /// 
        /// </summary>
        public double WeightedLeft { get { return m_weightedLeft; } }

        /// <summary>
        /// 
        /// </summary>
        public double WeightedRight { get { return m_weightedRight; } }

        public void Init(double[] uniqueTargets, double[] targets, double[] weights, Interval1D interval)
        {
            if (targets == null) { throw new ArgumentException("targets"); }
            if (weights == null) { throw new ArgumentException("weights"); }
            m_targets = targets;
            m_weights = weights;
            m_interval = interval;

            m_weightedTotal = 0.0;
            m_weightedLeft = 0.0;
            m_weightedRight = 0.0;

            m_meanLeft = 0.0;
            m_meanRight = 0.0;
            m_meanTotal = 0.0;

            m_sqSumLeft = 0.0;
            m_sqSumRight = 0.0;
            m_sqSumTotal = 0.0;

            m_varRight = 0.0;
            m_varLeft = 0.0;

            m_sumLeft = 0.0;
            m_sumRight = 0.0;
            m_sumTotal = 0.0;

            var w = 1.0;
            var weightsPresent = m_weights.Length != 0;

            for (int i = m_interval.FromInclusive; i < m_interval.ToExclusive; i++)
            {
                if (weightsPresent)
                    w = weights[i];

                var targetValue = targets[i];
                var wTarget = w * targetValue;
                m_sumTotal += wTarget;
                m_sqSumTotal += wTarget * targetValue;

                m_weightedTotal += w;
            }

            m_meanTotal = m_sumTotal / m_weightedTotal;

            m_currentPosition = m_interval.FromInclusive;
            this.Reset();
        }

        public void UpdateInterval(Interval1D newInterval)
        {
            m_interval = newInterval;
        }

        /// <summary>
        /// Sets the current index to <paramref name="newPosition"/>
        /// <c>AND</c> triggers the computation of KPIs.
        /// </summary>
        /// <param name="newPosition">The new split position.</param>
        public void UpdateIndex(int newPosition)
        {
            if (m_currentPosition > newPosition)
            {
                throw new ArgumentException("New position: " + newPosition +
                    " must be larget than current: " + m_currentPosition);
            }

            m_currentPosition = newPosition;
        }

        public void Reset()
        {
            m_currentPosition = m_interval.FromInclusive;

            m_weightedLeft = 0.0;
            m_weightedRight = m_weightedTotal;

            m_meanRight = m_meanTotal;
            m_meanLeft = 0.0;
            m_sumRight = m_sqSumTotal;
            m_sqSumLeft = 0.0;

            m_varRight = (m_sqSumRight / m_weightedTotal -
                m_meanRight * m_meanRight);
            m_varLeft = 0.0;

            m_sumRight = m_sumTotal;
            m_sumLeft = 0.0;

            this.FeatureSortedTopPerformers = null;
            this.TopThresoldValue = double.NaN;
        }

        public double ImpurityImprovement(double impurity)
        {
            throw new NotImplementedException();
        }

        public double NodeImpurity()
        {
            throw new NotImplementedException();
        }

        public ChildImpurities ChildImpurities()
        {
            throw new NotImplementedException();
        }

        public double LeafValue()
        {
            return m_meanTotal;
        }

        public double[] TargetNames
        {
            get { return new double[0]; }
        }

        public double[] LeafProbabilities()
        {
            return new double[0];
        }

        /// <summary>
        /// Sets the indicator for the T and U sets, and the current top threshold performance.
        /// It is assumed that <paramref name="featureSortedTopPerformers"/> is sorted in the same way as ITreeBuilder.m_workFeature
        /// </summary>
        /// <param name="featureSortedTopPerformers"></param>
        /// <param name="topThresholdValue"></param>
        public void SetTopPerformers(bool[] featureSortedTopPerformers, double topThresholdValue)
        {
            this.FeatureSortedTopPerformers = featureSortedTopPerformers;
            this.TopThresoldValue = topThresholdValue;
        }
    }
}

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
    public class TopPerformanceImpurityCalculator : IImpurityCalculator
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
            throw new NotImplementedException();
        }

        public void UpdateInterval(Interval1D newInterval)
        {
            throw new NotImplementedException();
        }

        public void UpdateIndex(int newPosition)
        {
            throw new NotImplementedException();
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
    }
}

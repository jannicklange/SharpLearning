using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    using SharpLearning.Containers.Extensions;
    using SharpLearning.Containers.Views;
    using SharpLearning.DecisionTrees.ImpurityCalculators;

    public class TopPerformanceResolutionSplitSearcher : ISplitSearcher
    {
        readonly int m_minimumSplitSize;
        readonly double m_minimumLeafWeight;

        /// <summary>
        /// Searches for the best split using a brute force approach on all unique threshold values. 
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        public TopPerformanceResolutionSplitSearcher(int minimumSplitSize)
            : this(minimumSplitSize, 0d)
        {
        }

        /// <summary>
        /// Searches for the best split using a brute force approach on all unique threshold values. 
        /// The implementation assumes that the features and targets have been sorted
        /// together using the features as sort criteria
        /// </summary>
        /// <param name="minimumSplitSize">The minimum size for a node to be split</param>
        /// <param name="minimumLeafWeight">Minimum leaf weight when splitting</param>
        public TopPerformanceResolutionSplitSearcher(int minimumSplitSize, double minimumLeafWeight)
        {
            if (minimumSplitSize <= 0) { throw new ArgumentException("minimum split size must be larger than 0"); }
            m_minimumSplitSize = minimumSplitSize;
            m_minimumLeafWeight = minimumLeafWeight;
        }

        /// <summary>
        /// TODO The find best split.
        /// </summary>
        /// <param name="impurityCalculator">
        /// TODO The impurity calculator.
        /// </param>
        /// <param name="feature">
        /// TODO The feature.
        /// </param>
        /// <param name="targets">
        /// TODO The targets.
        /// </param>
        /// <param name="parentInterval">
        /// TODO The parent interval.
        /// </param>
        /// <param name="parentImpurity">
        /// TODO The parent impurity.
        /// </param>
        /// <returns>
        /// The <see cref="SplitResult"/>.
        /// </returns>
        public SplitResult FindBestSplit(IImpurityCalculator impurityCalculatorArg, double[] feature, double[] targets, Interval1D parentInterval, double parentImpurity)
        {
            if (!(impurityCalculatorArg is ITopImpurityCalculator))
            {
                throw new ArgumentException($"Impurity Calculator needs to implement {typeof(ITopImpurityCalculator)}", nameof(impurityCalculatorArg));
            }

            var impurityCalculator = (ITopImpurityCalculator)impurityCalculatorArg;

            // determine number of top performers
            // Proportion of samples considered to be top performers // ToDo: config!
            var q = .1;
            var h = (int)Math.Ceiling(targets.Length * q);

            var vh = targets.NthSmallestElement(h - 1);
            // create indicator array
            var isTopPerformer = targets.Select(t => t <= vh).ToArray();

            // pass values to impurity calculator

            var bestSplitIndex = -1;
            var bestThreshold = 0.0;
            var bestImpurityImprovement = 0.0;
            var bestImpurityLeft = 0.0;
            var bestImpurityRight = 0.0;

            int prevSplit = parentInterval.FromInclusive;
            var prevValue = feature[prevSplit];
            var prevTarget = targets[prevSplit];
            
            impurityCalculator.SetTopPerformers(isTopPerformer, vh);
            impurityCalculator.UpdateInterval(parentInterval);

            for (int j = prevSplit + 1; j < parentInterval.ToExclusive; j++)
            {
                var currentValue = feature[j];
                var currentTarget = targets[j];
                if (Math.Abs(prevValue - currentValue) > 1e-10)
                {
                    var currentSplit = j;
                    var leftSize = (double)(currentSplit - parentInterval.FromInclusive);
                    var rightSize = (double)(parentInterval.ToExclusive - currentSplit);

                    if (Math.Min(leftSize, rightSize) >= m_minimumSplitSize)
                    {
                        impurityCalculator.UpdateIndex(currentSplit);

                        if (impurityCalculator.WeightedLeft < m_minimumLeafWeight ||
                            impurityCalculator.WeightedRight < m_minimumLeafWeight)
                        {
                            continue;
                        }

                        var improvement = impurityCalculator.ImpurityImprovement(parentImpurity);

                        if (improvement > bestImpurityImprovement)
                        {
                            var childImpurities = impurityCalculator.ChildImpurities(); // could be avoided

                            bestImpurityImprovement = improvement;
                            bestThreshold = (currentValue + prevValue) * 0.5;
                            //// ToDo: validate if behavhior changes if split is performed exactly on existing feature value
                            //bestThreshold = currentValue;
                            bestSplitIndex = currentSplit;
                            bestImpurityLeft = childImpurities.Left;
                            bestImpurityRight = childImpurities.Right;
                        }

                        prevSplit = j;
                    }
                }

                prevValue = currentValue;
                prevTarget = currentTarget;
            }

            return new SplitResult(bestSplitIndex, bestThreshold,
                bestImpurityImprovement, bestImpurityLeft, bestImpurityRight);
        }
    }
}

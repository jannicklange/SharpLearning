using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.SplitSearchers
{
    using SharpLearning.Containers.Views;
    using SharpLearning.DecisionTrees.ImpurityCalculators;

    public class TopPerformanceResolutionSplitSearcher : ISplitSearcher
    {
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
        public SplitResult FindBestSplit(IImpurityCalculator impurityCalculator, double[] feature, double[] targets, Interval1D parentInterval, double parentImpurity)
        {
            var result = new SplitResult();
            return result;
        }
    }
}

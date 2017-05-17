using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpLearning.DecisionTrees.ImpurityCalculators
{
    public interface ITopImpurityCalculator : IImpurityCalculator
    {
        void SetTopPerformers(bool[] featureSortedTopPerformers, double topThresholdValue);
    }
}

using SharpLearning.Containers.Views;
using SharpLearning.DecisionTrees.Nodes;

namespace SharpLearning.DecisionTrees.TreeBuilders
{
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.SplitSearchers;

    /// <summary>
    /// Tree builder interface
    /// </summary>
    public interface ITreeBuilder<TTreeType, TSplitSearcher, TImpurityCalculator> where TTreeType : BinaryTree
                                                                                where TSplitSearcher : ISplitSearcher<TImpurityCalculator>
                                                                                where TImpurityCalculator : IImpurityCalculator
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <param name="targets"></param>
        /// <param name="indices"></param>
        /// <param name="weights"></param>
        /// <returns></returns>
        TTreeType Build(F64MatrixView observations, double[] targets, int[] indices, double[] weights);
    }
}

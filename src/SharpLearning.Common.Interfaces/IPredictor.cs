
namespace SharpLearning.Common.Interfaces
{
    using SharpLearning.Containers.Matrices;

    /// <summary>
    /// General interface for predictor. 
    /// </summary>
    /// <typeparam name="TPrediction">The prediction type of the resulting model.</typeparam>
    public interface IPredictor<TPrediction>
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="observations"></param>
        /// <returns></returns>
        TPrediction Predict(double[] observations);

        TPrediction[] Predict(F64Matrix observation);

        TPrediction[] Predict(F64Matrix observations, int[] indices);
    }
}

using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.RandomForest.Learners;
using SharpLearning.RandomForest.Models;
using SharpLearning.RandomForest.Test.Properties;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.RandomForest.Test.Models
{
    using SharpLearning.DecisionTrees.ImpurityCalculators;
    using SharpLearning.DecisionTrees.SplitSearchers;

    [TestClass]
    public class RegressionForestModelTest
    {
        [TestMethod]
        public void RegressionForestModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.15381141277554411, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionForestModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.15381141277554411, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionForestModel_PredictCertainty_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = new CertaintyPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictCertainty(observations.Row(i));
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.15381141277554411, error, 0.0000001);

            var expected = new CertaintyPrediction[] { new CertaintyPrediction(0.379151515151515, 0.0608255007215007), new CertaintyPrediction(0.411071351850763, 0.0831655436577049), new CertaintyPrediction(0.243420918950331, 0.0452827034233046), new CertaintyPrediction(0.302332251082251, 0.0699917594408057), new CertaintyPrediction(0.411071351850763, 0.0831655436577049), new CertaintyPrediction(0.175743762773174, 0.0354069437824887), new CertaintyPrediction(0.574083361083361, 0.0765858693929188), new CertaintyPrediction(0.259063776093188, 0.0491198812971218), new CertaintyPrediction(0.163878898878899, 0.0331543420321184), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.418472943722944, 0.0607014359023913), new CertaintyPrediction(0.243420918950331, 0.0452827034233046), new CertaintyPrediction(0.443779942279942, 0.0941961872991865), new CertaintyPrediction(0.156999361749362, 0.0435804333960299), new CertaintyPrediction(0.591222034501446, 0.0873624628347336), new CertaintyPrediction(0.123822406351818, 0.0283119805431255), new CertaintyPrediction(0.162873993653405, 0.0333697457759022), new CertaintyPrediction(0.596261932511932, 0.0695341060210394), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.418472943722944, 0.0607014359023913), new CertaintyPrediction(0.329000027750028, 0.0788869852405852), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.499770375049787, 0.0913884936411888), new CertaintyPrediction(0.140025508804921, 0.0309875116490099), new CertaintyPrediction(0.161207326986739, 0.0336321035325246), new CertaintyPrediction(0.389553418803419, 0.0744433596104835), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionForestModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);
            var actual = sut.PredictCertainty(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.15381141277554411, error, 0.0000001);

            var expected = new CertaintyPrediction[] { new CertaintyPrediction(0.379151515151515, 0.0608255007215007), new CertaintyPrediction(0.411071351850763, 0.0831655436577049), new CertaintyPrediction(0.243420918950331, 0.0452827034233046), new CertaintyPrediction(0.302332251082251, 0.0699917594408057), new CertaintyPrediction(0.411071351850763, 0.0831655436577049), new CertaintyPrediction(0.175743762773174, 0.0354069437824887), new CertaintyPrediction(0.574083361083361, 0.0765858693929188), new CertaintyPrediction(0.259063776093188, 0.0491198812971218), new CertaintyPrediction(0.163878898878899, 0.0331543420321184), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.418472943722944, 0.0607014359023913), new CertaintyPrediction(0.243420918950331, 0.0452827034233046), new CertaintyPrediction(0.443779942279942, 0.0941961872991865), new CertaintyPrediction(0.156999361749362, 0.0435804333960299), new CertaintyPrediction(0.591222034501446, 0.0873624628347336), new CertaintyPrediction(0.123822406351818, 0.0283119805431255), new CertaintyPrediction(0.162873993653405, 0.0333697457759022), new CertaintyPrediction(0.596261932511932, 0.0695341060210394), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.418472943722944, 0.0607014359023913), new CertaintyPrediction(0.329000027750028, 0.0788869852405852), new CertaintyPrediction(0.671753996003996, 0.0624466591504497), new CertaintyPrediction(0.499770375049787, 0.0913884936411888), new CertaintyPrediction(0.140025508804921, 0.0309875116490099), new CertaintyPrediction(0.161207326986739, 0.0336321035325246), new CertaintyPrediction(0.389553418803419, 0.0744433596104835), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionForestModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { {"PreviousExperience_month", 100},
                {"AptitudeTestScore", 42.3879919692465 }};

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionForestModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(100, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 59.2053755086635, 139.67487667643803 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionForestModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionRandomForestLearner<OnlyUniqueThresholdsSplitSearcher<RegressionImpurityCalculator>, RegressionImpurityCalculator>(2, 5, 100, 1, 0.0001, 1.0, 42, false);
            var sut = learner.Learn(observations, targets);

            // save model.
            var writer = new StringWriter();
            sut.Save(() => writer);

            var modelString = writer.ToString();
            Assert.AreEqual(this.RegressionForestModelString, modelString);

            // load model and assert prediction results.
            sut = RegressionForestModel.Load(() => new StringReader(writer.ToString()));
            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var actual = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14547628738104926, actual, 0.0001);

        }

        [TestMethod]
        public void RegressionForestModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(this.RegressionForestModelString);
            var sut = RegressionForestModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14547628738104926, error, 0.0000001);
        }

        void Write(CertaintyPrediction[] predictions)
        {
            var value = "new CertaintyPrediction[] {";
            foreach (var item in predictions)
            {
                value += "new CertaintyPrediction(" + item.Prediction + ", " + item.Variance + "), ";
            }

            value += "};";

            Trace.WriteLine(value);
        }

        readonly string RegressionForestModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionForestModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.RandomForest.Models\">\r\n  <m_models xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:anyType z:Id=\"3\" xmlns:d3p1=\"SharpLearning.DecisionTrees.Models\" i:type=\"d3p1:RegressionDecisionTreeModel\">\r\n      <Nodes z:Id=\"4\" z:Size=\"7\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>6</RightIndex>\r\n          <Value>20</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>9.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.11111111111111111</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>0</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>4</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>5</RightIndex>\r\n          <Value>2</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.6</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>5</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.2857142857142857</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>6</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities z:Id=\"5\" z:Size=\"4\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:ArrayOfdouble z:Id=\"6\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n      </Probabilities>\r\n      <TargetNames z:Id=\"10\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:double>0</d2p1:double>\r\n        <d2p1:double>1</d2p1:double>\r\n      </TargetNames>\r\n      <VariableImportance z:Id=\"11\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:double>0.13296703296703297</d2p1:double>\r\n        <d2p1:double>2.448260073260073</d2p1:double>\r\n      </VariableImportance>\r\n    </d2p1:anyType>\r\n    <d2p1:anyType z:Id=\"12\" xmlns:d3p1=\"SharpLearning.DecisionTrees.Models\" i:type=\"d3p1:RegressionDecisionTreeModel\">\r\n      <Nodes z:Id=\"13\" z:Size=\"7\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>0</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>4</RightIndex>\r\n          <Value>3.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>13.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.42857142857142855</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>5</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>6</RightIndex>\r\n          <Value>17</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>5</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.6</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>6</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.5</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities z:Id=\"14\" z:Size=\"4\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:ArrayOfdouble z:Id=\"15\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"16\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"17\" z:Size=\"0\" />\r\n        <d2p1:ArrayOfdouble z:Id=\"18\" z:Size=\"0\" />\r\n      </Probabilities>\r\n      <TargetNames z:Id=\"19\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:double>0</d2p1:double>\r\n        <d2p1:double>1</d2p1:double>\r\n      </TargetNames>\r\n      <VariableImportance z:Id=\"20\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d2p1:double>0.757342657342657</d2p1:double>\r\n        <d2p1:double>0.40714285714285714</d2p1:double>\r\n      </VariableImportance>\r\n    </d2p1:anyType>\r\n  </m_models>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"2\">\r\n    <d2p1:double>0.89030969030969</d2p1:double>\r\n    <d2p1:double>2.85540293040293</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</RegressionForestModel>";

    }
}

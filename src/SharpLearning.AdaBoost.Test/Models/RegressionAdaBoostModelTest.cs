using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Models;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.AdaBoost.Test.Models
{
    [TestClass]
    public class RegressionAdaBoostModelTest
    {
        [TestMethod]
        public void RegressionAdaBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14185814185814186, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.14185814185814186, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 33.8004886838701 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void RegressionAdaBoostModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 5.9121817956101106, 17.49140922459933 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionAdaBoostModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new RegressionAdaBoostLearner(2);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            var actual = writer.ToString();
            Assert.AreEqual(RegressionDecisionTreeModelString, actual);
        }

        [TestMethod]
        public void RegressionAdaBoostModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(RegressionDecisionTreeModelString);
            var sut = RegressionAdaBoostModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanAbsolutErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.22527472527472531, error, 0.0000001);
        }

        readonly string RegressionDecisionTreeModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionAdaBoostModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.AdaBoost.Models\">\r\n  <m_modelWeights xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:double>1.0840134892469564</d2p1:double>\r\n    <d2p1:double>1.2719860712587807</d2p1:double>\r\n  </m_modelWeights>\r\n  <m_models xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"3\" z:Size=\"2\">\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"4\">\r\n      <Nodes z:Id=\"5\" z:Size=\"11\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>6</RightIndex>\r\n          <Value>13.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>2.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>0</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>4</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>5</RightIndex>\r\n          <Value>1.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.14285714285714285</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>5</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>0</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>7</LeftIndex>\r\n          <NodeIndex>6</NodeIndex>\r\n          <RightIndex>8</RightIndex>\r\n          <Value>3</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>7</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>9</LeftIndex>\r\n          <NodeIndex>8</NodeIndex>\r\n          <RightIndex>10</RightIndex>\r\n          <Value>17</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>4</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>9</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>5</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>10</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.5</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"6\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"10\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"11\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"12\" z:Size=\"0\" />\r\n      </Probabilities>\r\n      <TargetNames xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"13\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>1</d4p1:double>\r\n      </TargetNames>\r\n      <VariableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"14\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0.10622710622710622</d4p1:double>\r\n        <d4p1:double>3.6787585479535942</d4p1:double>\r\n      </VariableImportance>\r\n    </d2p1:RegressionDecisionTreeModel>\r\n    <d2p1:RegressionDecisionTreeModel z:Id=\"15\">\r\n      <Nodes z:Id=\"16\" z:Size=\"7\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>6</RightIndex>\r\n          <Value>20</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>4.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>4</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>5</RightIndex>\r\n          <Value>9.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>5</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0.21428571428571427</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>6</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"17\" z:Size=\"4\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:ArrayOfdouble z:Id=\"18\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"19\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"20\" z:Size=\"0\" />\r\n        <d4p1:ArrayOfdouble z:Id=\"21\" z:Size=\"0\" />\r\n      </Probabilities>\r\n      <TargetNames xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>1</d4p1:double>\r\n      </TargetNames>\r\n      <VariableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"23\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>2.9853081700907786</d4p1:double>\r\n      </VariableImportance>\r\n    </d2p1:RegressionDecisionTreeModel>\r\n  </m_models>\r\n  <m_predictions xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"24\" z:Size=\"2\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>0</d2p1:double>\r\n  </m_predictions>\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"25\" z:Size=\"2\">\r\n    <d2p1:double>0.11515161607385251</d2p1:double>\r\n    <d2p1:double>7.7850943004347517</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</RegressionAdaBoostModel>";
    }
}

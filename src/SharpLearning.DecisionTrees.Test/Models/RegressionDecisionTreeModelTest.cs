using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.Containers;
using SharpLearning.Containers.Extensions;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.DecisionTrees.Models;
using SharpLearning.DecisionTrees.Test.Properties;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using System.Collections.Generic;
using System.IO;

namespace SharpLearning.DecisionTrees.Test.Models
{
    [TestClass]
    public class RegressionDecisionTreeModelTest
    {
        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i)); 
            }

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Predict_Multiple_Indexed()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var indices = new int[] { 0, 3, 4, 5, 6, 7, 8, 9, 20, 21 };
            var predictions = sut.Predict(observations, indices);

            var indexedTargets = targets.GetIndices(indices);
            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(indexedTargets, predictions);

            Assert.AreEqual(0.023821615502626264, error, 0.0000001);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "F2", 100.0 }, { "F1", 0.0 } };

            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();
            var rows = targets.Length;
            var featureNameToIndex = new Dictionary<string, int> { { "F1", 0 }, { "F2", 1 } };

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.0, 364.56356850440511 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var learner = new RegressionDecisionTreeLearner(100, 2, 0.1, 42, 4);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);

            Assert.AreEqual(RegressionDecisionTreeModelString, writer.ToString());
        }

        [TestMethod]
        public void RegressionDecisionTreeModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.DecisionTreeData));
            var observations = parser.EnumerateRows("F1", "F2").ToF64Matrix();
            var targets = parser.EnumerateRows("T").ToF64Vector();

            var reader = new StringReader(RegressionDecisionTreeModelString);
            var sut = RegressionDecisionTreeModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new MeanSquaredErrorRegressionMetric();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.032120286249559482, error, 0.0000001);
        }

        readonly string RegressionDecisionTreeModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<RegressionDecisionTreeModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\">\r\n  <Nodes z:Id=\"2\" z:Size=\"25\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>1</LeftIndex>\r\n      <NodeIndex>0</NodeIndex>\r\n      <RightIndex>10</RightIndex>\r\n      <Value>0.397254</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>2</LeftIndex>\r\n      <NodeIndex>1</NodeIndex>\r\n      <RightIndex>5</RightIndex>\r\n      <Value>0.20301550000000002</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>3</LeftIndex>\r\n      <NodeIndex>2</NodeIndex>\r\n      <RightIndex>4</RightIndex>\r\n      <Value>0.14998250000000002</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>3</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>-0.054810500000000005</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>4</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>0.071894545454545447</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>6</LeftIndex>\r\n      <NodeIndex>5</NodeIndex>\r\n      <RightIndex>7</RightIndex>\r\n      <Value>0.3190235</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>6</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>1.094141117647059</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>8</LeftIndex>\r\n      <NodeIndex>7</NodeIndex>\r\n      <RightIndex>9</RightIndex>\r\n      <Value>0.3652945</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>8</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>0.80301928571428582</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>4</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>9</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>1.1078694999999998</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>11</LeftIndex>\r\n      <NodeIndex>10</NodeIndex>\r\n      <RightIndex>18</RightIndex>\r\n      <Value>0.59574250000000006</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>12</LeftIndex>\r\n      <NodeIndex>11</NodeIndex>\r\n      <RightIndex>13</RightIndex>\r\n      <Value>0.48716800000000005</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>5</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>12</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>1.88108975</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>14</LeftIndex>\r\n      <NodeIndex>13</NodeIndex>\r\n      <RightIndex>15</RightIndex>\r\n      <Value>0.5380695</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>6</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>14</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>2.0843065555555551</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>16</LeftIndex>\r\n      <NodeIndex>15</NodeIndex>\r\n      <RightIndex>17</RightIndex>\r\n      <Value>0.56256449999999991</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>7</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>16</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>1.81453875</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>8</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>17</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>2.072091</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>19</LeftIndex>\r\n      <NodeIndex>18</NodeIndex>\r\n      <RightIndex>24</RightIndex>\r\n      <Value>0.8071625</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>20</LeftIndex>\r\n      <NodeIndex>19</NodeIndex>\r\n      <RightIndex>21</RightIndex>\r\n      <Value>0.6214955</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>9</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>20</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>3.1442664000000002</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>1</FeatureIndex>\r\n      <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n      <LeftIndex>22</LeftIndex>\r\n      <NodeIndex>21</NodeIndex>\r\n      <RightIndex>23</RightIndex>\r\n      <Value>0.66172000000000009</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>10</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>22</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>2.8252029999999997</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>11</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>23</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>2.9832834545454552</Value>\r\n    </Node>\r\n    <Node>\r\n      <FeatureIndex>-1</FeatureIndex>\r\n      <LeafProbabilityIndex>12</LeafProbabilityIndex>\r\n      <LeftIndex>-1</LeftIndex>\r\n      <NodeIndex>24</NodeIndex>\r\n      <RightIndex>-1</RightIndex>\r\n      <Value>3.9871632</Value>\r\n    </Node>\r\n  </Nodes>\r\n  <Probabilities xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"3\" z:Size=\"13\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n    <d2p1:ArrayOfdouble z:Id=\"4\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"5\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"6\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"7\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"8\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"9\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"10\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"11\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"12\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"13\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"14\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"15\" z:Size=\"0\" />\r\n    <d2p1:ArrayOfdouble z:Id=\"16\" z:Size=\"0\" />\r\n  </Probabilities>\r\n  <TargetNames xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"17\" z:Size=\"200\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n    <d2p1:double>1.88318</d2p1:double>\r\n    <d2p1:double>0.063908</d2p1:double>\r\n    <d2p1:double>3.042257</d2p1:double>\r\n    <d2p1:double>2.305004</d2p1:double>\r\n    <d2p1:double>-0.067698</d2p1:double>\r\n    <d2p1:double>1.662809</d2p1:double>\r\n    <d2p1:double>3.275749</d2p1:double>\r\n    <d2p1:double>1.118077</d2p1:double>\r\n    <d2p1:double>2.095059</d2p1:double>\r\n    <d2p1:double>1.181912</d2p1:double>\r\n    <d2p1:double>0.221663</d2p1:double>\r\n    <d2p1:double>0.938453</d2p1:double>\r\n    <d2p1:double>4.149409</d2p1:double>\r\n    <d2p1:double>3.105444</d2p1:double>\r\n    <d2p1:double>1.896278</d2p1:double>\r\n    <d2p1:double>-0.121345</d2p1:double>\r\n    <d2p1:double>3.161652</d2p1:double>\r\n    <d2p1:double>4.135358</d2p1:double>\r\n    <d2p1:double>0.859063</d2p1:double>\r\n    <d2p1:double>1.170272</d2p1:double>\r\n    <d2p1:double>1.68796</d2p1:double>\r\n    <d2p1:double>1.979745</d2p1:double>\r\n    <d2p1:double>0.06869</d2p1:double>\r\n    <d2p1:double>4.052137</d2p1:double>\r\n    <d2p1:double>3.156316</d2p1:double>\r\n    <d2p1:double>2.95063</d2p1:double>\r\n    <d2p1:double>0.068935</d2p1:double>\r\n    <d2p1:double>2.85402</d2p1:double>\r\n    <d2p1:double>0.999743</d2p1:double>\r\n    <d2p1:double>4.048082</d2p1:double>\r\n    <d2p1:double>0.230923</d2p1:double>\r\n    <d2p1:double>0.816693</d2p1:double>\r\n    <d2p1:double>0.130713</d2p1:double>\r\n    <d2p1:double>-0.537706</d2p1:double>\r\n    <d2p1:double>-0.339109</d2p1:double>\r\n    <d2p1:double>2.124538</d2p1:double>\r\n    <d2p1:double>2.708292</d2p1:double>\r\n    <d2p1:double>4.01739</d2p1:double>\r\n    <d2p1:double>4.004021</d2p1:double>\r\n    <d2p1:double>1.022555</d2p1:double>\r\n    <d2p1:double>3.657442</d2p1:double>\r\n    <d2p1:double>4.073619</d2p1:double>\r\n    <d2p1:double>0.011994</d2p1:double>\r\n    <d2p1:double>3.640429</d2p1:double>\r\n    <d2p1:double>1.808497</d2p1:double>\r\n    <d2p1:double>1.431404</d2p1:double>\r\n    <d2p1:double>3.935544</d2p1:double>\r\n    <d2p1:double>1.162152</d2p1:double>\r\n    <d2p1:double>-0.22733</d2p1:double>\r\n    <d2p1:double>-0.068728</d2p1:double>\r\n    <d2p1:double>0.825051</d2p1:double>\r\n    <d2p1:double>2.008645</d2p1:double>\r\n    <d2p1:double>0.664566</d2p1:double>\r\n    <d2p1:double>4.180202</d2p1:double>\r\n    <d2p1:double>0.864845</d2p1:double>\r\n    <d2p1:double>1.851174</d2p1:double>\r\n    <d2p1:double>2.761993</d2p1:double>\r\n    <d2p1:double>4.075914</d2p1:double>\r\n    <d2p1:double>0.110229</d2p1:double>\r\n    <d2p1:double>2.181987</d2p1:double>\r\n    <d2p1:double>3.152528</d2p1:double>\r\n    <d2p1:double>3.046564</d2p1:double>\r\n    <d2p1:double>0.128954</d2p1:double>\r\n    <d2p1:double>1.062726</d2p1:double>\r\n    <d2p1:double>3.651742</d2p1:double>\r\n    <d2p1:double>4.40195</d2p1:double>\r\n    <d2p1:double>3.022888</d2p1:double>\r\n    <d2p1:double>2.874917</d2p1:double>\r\n    <d2p1:double>2.946801</d2p1:double>\r\n    <d2p1:double>2.893644</d2p1:double>\r\n    <d2p1:double>0.072131</d2p1:double>\r\n    <d2p1:double>1.748275</d2p1:double>\r\n    <d2p1:double>1.912047</d2p1:double>\r\n    <d2p1:double>3.710686</d2p1:double>\r\n    <d2p1:double>1.719148</d2p1:double>\r\n    <d2p1:double>2.17409</d2p1:double>\r\n    <d2p1:double>3.656357</d2p1:double>\r\n    <d2p1:double>3.522504</d2p1:double>\r\n    <d2p1:double>2.234126</d2p1:double>\r\n    <d2p1:double>1.859772</d2p1:double>\r\n    <d2p1:double>2.097017</d2p1:double>\r\n    <d2p1:double>0.001794</d2p1:double>\r\n    <d2p1:double>1.231928</d2p1:double>\r\n    <d2p1:double>2.953862</d2p1:double>\r\n    <d2p1:double>-0.116803</d2p1:double>\r\n    <d2p1:double>2.638864</d2p1:double>\r\n    <d2p1:double>3.943428</d2p1:double>\r\n    <d2p1:double>-0.328513</d2p1:double>\r\n    <d2p1:double>-0.099866</d2p1:double>\r\n    <d2p1:double>-0.030836</d2p1:double>\r\n    <d2p1:double>2.359786</d2p1:double>\r\n    <d2p1:double>3.212581</d2p1:double>\r\n    <d2p1:double>0.188975</d2p1:double>\r\n    <d2p1:double>1.904109</d2p1:double>\r\n    <d2p1:double>3.007114</d2p1:double>\r\n    <d2p1:double>3.845834</d2p1:double>\r\n    <d2p1:double>3.079411</d2p1:double>\r\n    <d2p1:double>1.939739</d2p1:double>\r\n    <d2p1:double>2.880078</d2p1:double>\r\n    <d2p1:double>3.063577</d2p1:double>\r\n    <d2p1:double>4.116296</d2p1:double>\r\n    <d2p1:double>-0.240963</d2p1:double>\r\n    <d2p1:double>4.066299</d2p1:double>\r\n    <d2p1:double>4.011834</d2p1:double>\r\n    <d2p1:double>0.07771</d2p1:double>\r\n    <d2p1:double>3.103069</d2p1:double>\r\n    <d2p1:double>2.811897</d2p1:double>\r\n    <d2p1:double>-0.10463</d2p1:double>\r\n    <d2p1:double>0.025216</d2p1:double>\r\n    <d2p1:double>4.330063</d2p1:double>\r\n    <d2p1:double>3.087091</d2p1:double>\r\n    <d2p1:double>2.269988</d2p1:double>\r\n    <d2p1:double>4.010701</d2p1:double>\r\n    <d2p1:double>3.119542</d2p1:double>\r\n    <d2p1:double>3.723411</d2p1:double>\r\n    <d2p1:double>2.792078</d2p1:double>\r\n    <d2p1:double>2.192787</d2p1:double>\r\n    <d2p1:double>2.081305</d2p1:double>\r\n    <d2p1:double>1.714463</d2p1:double>\r\n    <d2p1:double>0.885854</d2p1:double>\r\n    <d2p1:double>1.028187</d2p1:double>\r\n    <d2p1:double>1.951497</d2p1:double>\r\n    <d2p1:double>1.709427</d2p1:double>\r\n    <d2p1:double>0.144068</d2p1:double>\r\n    <d2p1:double>3.88024</d2p1:double>\r\n    <d2p1:double>0.921876</d2p1:double>\r\n    <d2p1:double>1.979316</d2p1:double>\r\n    <d2p1:double>3.085768</d2p1:double>\r\n    <d2p1:double>3.476122</d2p1:double>\r\n    <d2p1:double>3.993679</d2p1:double>\r\n    <d2p1:double>3.07788</d2p1:double>\r\n    <d2p1:double>-0.105365</d2p1:double>\r\n    <d2p1:double>-0.164703</d2p1:double>\r\n    <d2p1:double>1.096814</d2p1:double>\r\n    <d2p1:double>3.092879</d2p1:double>\r\n    <d2p1:double>2.987926</d2p1:double>\r\n    <d2p1:double>2.061264</d2p1:double>\r\n    <d2p1:double>2.746854</d2p1:double>\r\n    <d2p1:double>0.71671</d2p1:double>\r\n    <d2p1:double>0.103831</d2p1:double>\r\n    <d2p1:double>0.023776</d2p1:double>\r\n    <d2p1:double>-0.033299</d2p1:double>\r\n    <d2p1:double>1.942286</d2p1:double>\r\n    <d2p1:double>-0.006338</d2p1:double>\r\n    <d2p1:double>3.808753</d2p1:double>\r\n    <d2p1:double>0.652799</d2p1:double>\r\n    <d2p1:double>4.053747</d2p1:double>\r\n    <d2p1:double>4.56929</d2p1:double>\r\n    <d2p1:double>-0.032773</d2p1:double>\r\n    <d2p1:double>2.066236</d2p1:double>\r\n    <d2p1:double>0.222785</d2p1:double>\r\n    <d2p1:double>1.089268</d2p1:double>\r\n    <d2p1:double>1.487788</d2p1:double>\r\n    <d2p1:double>2.852033</d2p1:double>\r\n    <d2p1:double>0.024486</d2p1:double>\r\n    <d2p1:double>3.73775</d2p1:double>\r\n    <d2p1:double>0.045017</d2p1:double>\r\n    <d2p1:double>0.001238</d2p1:double>\r\n    <d2p1:double>3.892763</d2p1:double>\r\n    <d2p1:double>2.819376</d2p1:double>\r\n    <d2p1:double>2.830665</d2p1:double>\r\n    <d2p1:double>0.234633</d2p1:double>\r\n    <d2p1:double>1.810782</d2p1:double>\r\n    <d2p1:double>4.237235</d2p1:double>\r\n    <d2p1:double>3.034768</d2p1:double>\r\n    <d2p1:double>1.742106</d2p1:double>\r\n    <d2p1:double>1.16925</d2p1:double>\r\n    <d2p1:double>0.831165</d2p1:double>\r\n    <d2p1:double>3.729376</d2p1:double>\r\n    <d2p1:double>1.823205</d2p1:double>\r\n    <d2p1:double>4.02197</d2p1:double>\r\n    <d2p1:double>1.262939</d2p1:double>\r\n    <d2p1:double>4.159518</d2p1:double>\r\n    <d2p1:double>2.039114</d2p1:double>\r\n    <d2p1:double>4.101837</d2p1:double>\r\n    <d2p1:double>2.778672</d2p1:double>\r\n    <d2p1:double>1.228284</d2p1:double>\r\n    <d2p1:double>1.73662</d2p1:double>\r\n    <d2p1:double>-0.195046</d2p1:double>\r\n    <d2p1:double>-0.063215</d2p1:double>\r\n    <d2p1:double>3.305268</d2p1:double>\r\n    <d2p1:double>2.063627</d2p1:double>\r\n    <d2p1:double>0.89884</d2p1:double>\r\n    <d2p1:double>2.701692</d2p1:double>\r\n    <d2p1:double>1.992909</d2p1:double>\r\n    <d2p1:double>3.811393</d2p1:double>\r\n    <d2p1:double>4.353857</d2p1:double>\r\n    <d2p1:double>2.635641</d2p1:double>\r\n    <d2p1:double>2.856311</d2p1:double>\r\n    <d2p1:double>1.352682</d2p1:double>\r\n    <d2p1:double>2.336459</d2p1:double>\r\n    <d2p1:double>2.111651</d2p1:double>\r\n    <d2p1:double>0.121726</d2p1:double>\r\n    <d2p1:double>3.264605</d2p1:double>\r\n    <d2p1:double>2.103446</d2p1:double>\r\n    <d2p1:double>0.896855</d2p1:double>\r\n    <d2p1:double>4.22085</d2p1:double>\r\n    <d2p1:double>-0.217283</d2p1:double>\r\n    <d2p1:double>-0.300577</d2p1:double>\r\n    <d2p1:double>0.006014</d2p1:double>\r\n  </TargetNames>\r\n  <VariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"18\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>364.56356850440511</d2p1:double>\r\n  </VariableImportance>\r\n</RegressionDecisionTreeModel>";
    }
}

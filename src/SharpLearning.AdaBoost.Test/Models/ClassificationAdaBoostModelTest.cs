﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.AdaBoost.Models;
using SharpLearning.AdaBoost.Test.Properties;
using SharpLearning.Containers;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Classification;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace SharpLearning.AdaBoost.Test.Models
{
    [TestClass]
    public class ClassificationAdaBoostModelTest
    {
        [TestMethod]
        public void ClassificationAdaBoostModel_Predict_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = new double[rows];
            for (int i = 0; i < rows; i++)
            {
                predictions[i] = sut.Predict(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_Precit_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10);
            var sut = learner.Learn(observations, targets);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_PredictProbability_Single()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = new ProbabilityPrediction[rows];
            for (int i = 0; i < rows; i++)
            {
                actual[i] = sut.PredictProbability(observations.Row(i));
            }

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] {new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.553917222019051}, {1, 0.446082777980949}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.455270122123639}, {1, 0.544729877876361}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.590671208378385}, {1, 0.409328791621616}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.564961572849738}, {1, 0.435038427150263}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.455270122123639}, {1, 0.544729877876361}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.417527839140627}, {1, 0.582472160859373}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.409988559960094}, {1, 0.590011440039906}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.461264944069783}, {1, 0.538735055930217}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.590671208378385}, {1, 0.409328791621616}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549503146925505}, {1, 0.450496853074495}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.537653803214063}, {1, 0.462346196785938}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.37650723540928}, {1, 0.62349276459072}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.573579890413618}, {1, 0.426420109586382}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.524371409810479}, {1, 0.475628590189522}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.471117379964633}, {1, 0.528882620035367}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.436954866525023}, {1, 0.563045133474978}, }),new ProbabilityPrediction(1, new Dictionary<double, double> {{0, 0.404976804073458}, {1, 0.595023195926542}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.573579890413618}, {1, 0.426420109586382}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.549970403132686}, {1, 0.450029596867314}, }),new ProbabilityPrediction(0, new Dictionary<double, double> {{0, 0.630894242807786}, {1, 0.369105757192214}, }),};
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_PredictProbability_Multiple()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var rows = targets.Length;

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.PredictProbability(observations);
            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, actual.Select(p => p.Prediction).ToArray());

            Assert.AreEqual(0.038461538461538464, error, 0.0000001);

            var expected = new ProbabilityPrediction[] { new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.553917222019051 }, { 1, 0.446082777980949 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.455270122123639 }, { 1, 0.544729877876361 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.590671208378385 }, { 1, 0.409328791621616 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.564961572849738 }, { 1, 0.435038427150263 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.455270122123639 }, { 1, 0.544729877876361 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.417527839140627 }, { 1, 0.582472160859373 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.409988559960094 }, { 1, 0.590011440039906 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.461264944069783 }, { 1, 0.538735055930217 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.590671208378385 }, { 1, 0.409328791621616 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549503146925505 }, { 1, 0.450496853074495 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.537653803214063 }, { 1, 0.462346196785938 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.37650723540928 }, { 1, 0.62349276459072 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.573579890413618 }, { 1, 0.426420109586382 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.524371409810479 }, { 1, 0.475628590189522 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.471117379964633 }, { 1, 0.528882620035367 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.436954866525023 }, { 1, 0.563045133474978 }, }), new ProbabilityPrediction(1, new Dictionary<double, double> { { 0, 0.404976804073458 }, { 1, 0.595023195926542 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.573579890413618 }, { 1, 0.426420109586382 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.549970403132686 }, { 1, 0.450029596867314 }, }), new ProbabilityPrediction(0, new Dictionary<double, double> { { 0, 0.630894242807786 }, { 1, 0.369105757192214 }, }), };
            CollectionAssert.AreEqual(expected, actual);
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_GetVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();
            var featureNameToIndex = new Dictionary<string, int> { { "AptitudeTestScore", 0 }, 
                { "PreviousExperience_month", 1 } };

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetVariableImportance(featureNameToIndex);
            var expected = new Dictionary<string, double> { { "PreviousExperience_month", 100.0 }, 
                { "AptitudeTestScore", 24.0268096428771 } };

            Assert.AreEqual(expected.Count, actual.Count);
            var zip = expected.Zip(actual, (e, a) => new { Expected = e, Actual = a });

            foreach (var item in zip)
            {
                Assert.AreEqual(item.Expected.Key, item.Actual.Key);
                Assert.AreEqual(item.Expected.Value, item.Actual.Value, 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_GetRawVariableImportance()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationAdaBoostLearner(10, 1, 3);
            var sut = learner.Learn(observations, targets);

            var actual = sut.GetRawVariableImportance();
            var expected = new double[] { 0.65083327864662022, 2.7087794356399844 };

            Assert.AreEqual(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                Assert.AreEqual(expected[i], actual[i], 0.000001);
            }
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_Save()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var learner = new ClassificationAdaBoostLearner(2);
            var sut = learner.Learn(observations, targets);

            var writer = new StringWriter();
            sut.Save(() => writer);
            
            Assert.AreEqual(ClassificationAdaBoostModelString, writer.ToString());
        }

        [TestMethod]
        public void ClassificationAdaBoostModel_Load()
        {
            var parser = new CsvParser(() => new StringReader(Resources.AptitudeData));
            var observations = parser.EnumerateRows(v => v != "Pass").ToF64Matrix();
            var targets = parser.EnumerateRows("Pass").ToF64Vector();

            var reader = new StringReader(ClassificationAdaBoostModelString);
            var sut = ClassificationAdaBoostModel.Load(() => reader);

            var predictions = sut.Predict(observations);

            var evaluator = new TotalErrorClassificationMetric<double>();
            var error = evaluator.Error(targets, predictions);

            Assert.AreEqual(0.19230769230769232, error, 0.0000001);
        }


        void Write(ProbabilityPrediction[] predictions)
        {
            var value = "new ProbabilityPrediction[] {";
            foreach (var item in predictions)
            {
                value += "new ProbabilityPrediction(" + item.Prediction + ", new Dictionary<double, double> {";
                foreach (var prob in item.Probabilities)
                {
                    value += "{" + prob.Key + ", " + prob.Value + "}, ";
                }
                value += "}),";
            }
            value += "};";

            Trace.WriteLine(value);
        }

        readonly string ClassificationAdaBoostModelString =
            "<?xml version=\"1.0\" encoding=\"utf-16\"?>\r\n<ClassificationAdaBoostModel xmlns:i=\"http://www.w3.org/2001/XMLSchema-instance\" z:Id=\"1\" xmlns:z=\"http://schemas.microsoft.com/2003/10/Serialization/\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.AdaBoost.Models\">\r\n  <m_modelWeights xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"2\" z:Size=\"2\">\r\n    <d2p1:double>1.4350845252893221</d2p1:double>\r\n    <d2p1:double>1.4163266482187662</d2p1:double>\r\n  </m_modelWeights>\r\n  <m_models xmlns:d2p1=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Models\" z:Id=\"3\" z:Size=\"2\">\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"4\">\r\n      <Nodes z:Id=\"5\" z:Size=\"5\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>4</RightIndex>\r\n          <Value>20</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>2.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"6\" z:Size=\"3\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:ArrayOfdouble z:Id=\"7\" z:Size=\"2\">\r\n          <d4p1:double>0.49056603773584911</d4p1:double>\r\n          <d4p1:double>0.509433962264151</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n        <d4p1:ArrayOfdouble z:Id=\"8\" z:Size=\"2\">\r\n          <d4p1:double>0.57534246575342463</d4p1:double>\r\n          <d4p1:double>0.42465753424657537</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n        <d4p1:ArrayOfdouble z:Id=\"9\" z:Size=\"2\">\r\n          <d4p1:double>0.4642857142857143</d4p1:double>\r\n          <d4p1:double>0.5357142857142857</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n      </Probabilities>\r\n      <TargetNames xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"10\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>1</d4p1:double>\r\n      </TargetNames>\r\n      <VariableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"11\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>0.18033248802479651</d4p1:double>\r\n      </VariableImportance>\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n    <d2p1:ClassificationDecisionTreeModel z:Id=\"12\">\r\n      <Nodes z:Id=\"13\" z:Size=\"7\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>1</LeftIndex>\r\n          <NodeIndex>0</NodeIndex>\r\n          <RightIndex>4</RightIndex>\r\n          <Value>13.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>2</LeftIndex>\r\n          <NodeIndex>1</NodeIndex>\r\n          <RightIndex>3</RightIndex>\r\n          <Value>4.5</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>0</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>2</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>1</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>3</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>0</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>1</FeatureIndex>\r\n          <LeafProbabilityIndex>-1</LeafProbabilityIndex>\r\n          <LeftIndex>5</LeftIndex>\r\n          <NodeIndex>4</NodeIndex>\r\n          <RightIndex>6</RightIndex>\r\n          <Value>20</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>2</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>5</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n        <Node>\r\n          <FeatureIndex>-1</FeatureIndex>\r\n          <LeafProbabilityIndex>3</LeafProbabilityIndex>\r\n          <LeftIndex>-1</LeftIndex>\r\n          <NodeIndex>6</NodeIndex>\r\n          <RightIndex>-1</RightIndex>\r\n          <Value>1</Value>\r\n        </Node>\r\n      </Nodes>\r\n      <Probabilities xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"14\" z:Size=\"4\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:ArrayOfdouble z:Id=\"15\" z:Size=\"2\">\r\n          <d4p1:double>0.46391752577319589</d4p1:double>\r\n          <d4p1:double>0.53608247422680411</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n        <d4p1:ArrayOfdouble z:Id=\"16\" z:Size=\"2\">\r\n          <d4p1:double>0.55718475073313778</d4p1:double>\r\n          <d4p1:double>0.44281524926686211</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n        <d4p1:ArrayOfdouble z:Id=\"17\" z:Size=\"2\">\r\n          <d4p1:double>0.42899408284023677</d4p1:double>\r\n          <d4p1:double>0.57100591715976334</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n        <d4p1:ArrayOfdouble z:Id=\"18\" z:Size=\"2\">\r\n          <d4p1:double>0.4642857142857143</d4p1:double>\r\n          <d4p1:double>0.5357142857142857</d4p1:double>\r\n        </d4p1:ArrayOfdouble>\r\n      </Probabilities>\r\n      <TargetNames xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"19\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>1</d4p1:double>\r\n      </TargetNames>\r\n      <VariableImportance xmlns:d4p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"20\" z:Size=\"2\" xmlns=\"http://schemas.datacontract.org/2004/07/SharpLearning.DecisionTrees.Nodes\">\r\n        <d4p1:double>0</d4p1:double>\r\n        <d4p1:double>0.17822779700804667</d4p1:double>\r\n      </VariableImportance>\r\n    </d2p1:ClassificationDecisionTreeModel>\r\n  </m_models>\r\n  <m_predictions xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"21\" z:Size=\"0\" />\r\n  <m_rawVariableImportance xmlns:d2p1=\"http://schemas.microsoft.com/2003/10/Serialization/Arrays\" z:Id=\"22\" z:Size=\"2\">\r\n    <d2p1:double>0</d2p1:double>\r\n    <d2p1:double>0.51122114132712881</d2p1:double>\r\n  </m_rawVariableImportance>\r\n</ClassificationAdaBoostModel>";
    }
}

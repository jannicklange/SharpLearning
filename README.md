
SharpLearning Fork
=================

--------------
_**Note:**_

This is a fork of the [SharpLearning](https://github.com/mdabros/SharpLearning) project by Mads Dabros that contains some changes to the API of the RandomForest and TreeLearner classes. The classes/interfaces are extended in a way that allows for the implementation of a custom performance-focues split criterion, which is in the following paper: [GGA++](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/viewPaper/11435). <br/>
The fork is based on the [SharpLearning 0.25.04](https://www.nuget.org/packages/SharpLearning.RandomForest/0.25.0.4) Release on [nuget.org](nuget.org).

This fork is available under the [MIT License](LICENSE.md), just as the original project.

--------------

SharpLearning is an opensource machine learning library for C# .Net. 
The goal of SharpLearning is to provide .Net developers with easy access to machine learning algorithms and models.

Currently the main focus is supervised learning for classification and regression, 
while also providing the necesarry tools for optimizing and validating the trained models.

SharpLearning provides a simple high-level interface for machine learning algorithms.    
In SharpLearning a machine learning algorithm is refered to as a *Learner*, 
and a machine learning model is refered to as a *PredictorModel*. An example of usage can be seen below:

```c#
// Create a random forest learner for classification with 100 trees
var learner = new ClassificationRandomForestLearner(trees: 100);

// learn the model
var model = learner.Learn(observations, targets);

// use the model for predicting new observations
var predictions = model.Predict(testObservations);

// save the model for use with another application
model.Save(() => new StreamWriter("randomforest.xml"));
```

All machine learning algorithms and models implement the same interface for easy replacement.

Currently SharpLearning supports the following machine learning algorithms and models:

* DecisionTrees
* Adaboost (trees)
* GradientBoost (trees)
* RandomForest
* ExtraTrees
* NeuralNets (layers for fully connected and convolutional nets)
* Ensemble Learning

All the machine learning algorithms have sensible default hyperparameters for easy usage. 
However, several optimization methods are availible for hyperparameter tuning:

* GridSearch
* RandomSearch
* ParticleSwarm
* GlobalizedBoundedNelderMead
* SequentialModelBased  

License
-------

SharpLearning is covered under the terms of the [MIT](LICENSE.md) license. You may therefore link to it and use it in both opensource and proprietary software projects.

Documentation
-------------
SharpLearning contains xml documentation to help guide the user while using the library. 

Code examples and more information about how to use SharpLearning can be found in [SharpLearning.Examples](https://github.com/mdabros/SharpLearning.Examples)

The wiki also contains a set of guides on how to get started: 
 - [**Getting Started**](https://github.com/mdabros/SharpLearning/wiki/Getting-started).
 - [**Introduction to SharpLearning**](https://github.com/mdabros/SharpLearning/wiki/Introduction-to-SharpLearning).
 - [**Tuning Hyperparameters**](https://github.com/mdabros/SharpLearning/wiki/hyperparameter-tuning).


Installation
------------

The recommended way to get SharpLearning is to use NuGet. The packages are provided and maintained in the public [NuGet Gallery](https://nuget.org/profiles/mdabros/).
More information can be found in the [getting started](https://github.com/mdabros/SharpLearning/wiki/Getting-started) guide on the wiki

Learner and model packages:

- **SharpLearning.DecisionTrees** - Provides learning algorithms and models for DecisionTree regression and classification.
- **SharpLearning.AdaBoost** - Provides learning algorithms and models for AdaBoost regression and classification.
- **SharpLearning.RandomForest** - Provides learning algorithms and models for RandomForest and ExtraTrees regression and classification.
- **SharpLearning.GradientBoost** - Provides learning algorithms and models for GradientBoost regression and classification.
- **SharpLearning.Neural** - Provides learning algorithms and models for neural net regression and classification. Layers availible for fully connected and covolutional nets.
- **SharpLearning.Ensemble** - Provides ensemble learning for regression and classification. Makes it possible to combine the other learners/models from SharpLearning.
- **SharpLearning.Common.Interfaces** - Provides common interfaces for SharpLearning.

Validation and model selection packages:

- **SharpLearning.CrossValidation** - Provides cross-validation, training/test set samplers and learning curves for SharpLearning.
- **SharpLearning.Metrics** - Provides classification, regression, impurity and ranking metrics..
- **SharpLearning.Optimization** - Provides optimization algorithms for hyperparameter tuning.

Container/IO packages:

- **SharpLearning.Containers** - Provides containers and base extension methods for SharpLearning.
- **SharpLearning.InputOutput** - Provides csv parsing and serialization for SharpLearning.
- **SharpLearning.FeatureTransformations** - Provides CsvRow transforms like missing value replacement and matrix transforms like MinMaxNormalization.



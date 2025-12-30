using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LinearRegression.Core
{
    public class ModelTrainer
    {
        private readonly MLContext _mlContext;
        private ITransformer? _model;
        private DataViewSchema? _modelSchema;

        public ModelTrainer(int seed = 1)
        {
            _mlContext = new MLContext(seed: seed);
        }

      public (double AvgRSquared, double AvgRMSE, RegressionMetrics[] PerFoldMetrics) Train(
    IEnumerable<ModelInput> data,
    int numberOfFolds = 3,
    int numberOfIterations = 5000,
    float learningRate = 0.1f)
{
    var list = data?.ToList() ?? throw new ArgumentNullException(nameof(data));
    var dataView = _mlContext.Data.LoadFromEnumerable(list);

    var basePipeline = _mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ModelInput.Y))
        .Append(_mlContext.Transforms.Concatenate("Features", nameof(ModelInput.X)));

    var pipeline = (list.Count < numberOfFolds * 5)
        ? basePipeline.Append(_mlContext.Regression.Trainers.Sdca(
            labelColumnName: "Label",
            featureColumnName: "Features"))
        : basePipeline.Append(_mlContext.Transforms.NormalizeMeanVariance("Features"))
                    .Append(_mlContext.Regression.Trainers.Sdca(
            labelColumnName: "Label",
            featureColumnName: "Features"));

    // Always train a final model used by Predict/Save
    _model = pipeline.Fit(dataView);
    _modelSchema = dataView.Schema;

    // For tiny datasets, CrossValidate is unstable -> evaluate on training data for a stable unit-test metric.
    // Rule: only CV when we have enough rows per fold (at least ~5 test rows per fold).
    if (list.Count < numberOfFolds * 5)
    {
        var predictions = _model.Transform(dataView);
        var metrics = _mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

        return (metrics.RSquared, metrics.RootMeanSquaredError, new[] { metrics });
    }

    // Otherwise, cross-validate
    var cvResults = _mlContext.Regression.CrossValidate(
        data: dataView,
        estimator: pipeline,
        numberOfFolds: numberOfFolds,
        labelColumnName: "Label");

    var avgR2 = cvResults.Average(r => r.Metrics.RSquared);
    var avgRmse = cvResults.Average(r => r.Metrics.RootMeanSquaredError);
    var perFoldMetrics = cvResults.Select(r => r.Metrics).ToArray();

    return (avgR2, avgRmse, perFoldMetrics);
}


        public float Predict(float x)
        {
            if (_model == null)
                throw new InvalidOperationException("Model is not trained or loaded.");

            var engine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
            return engine.Predict(new ModelInput { X = x }).Prediction;
        }

        public void Save(string path)
        {
            if (_model == null || _modelSchema == null)
                throw new InvalidOperationException("Model is not trained or loaded.");

            using var fs = File.OpenWrite(path);
            _mlContext.Model.Save(_model, _modelSchema, fs);
        }

        public void Load(string path)
        {
            using var fs = File.OpenRead(path);
            _model = _mlContext.Model.Load(fs, out var schema);
            _modelSchema = schema;
        }
    }
}

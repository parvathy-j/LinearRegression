using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LinearRegression
{
    // Input data structure
    // =========================
    // DATA MODELS
    // =========================

    // Represents one row of input data
    // X = feature (independent variable)
    // Y = label (dependent variable / target)
    public class ModelInput
    {
        public float X { get; set; }
        public float Y { get; set; }
    }

    // Output prediction
    public class ModelOutput
    {
        [ColumnName("Score")]
        public float Prediction { get; set; }
    }

    class Program
    {
        static void Main()
        {
            // Create ML context
            // MLContext is the starting point for all ML.NET operations.
            // Providing a seed makes training and data splits reproducible.
            var mlContext = new MLContext(seed: 1);
            // Sample dataset following the regression model y = 2x + 3
            var trainingData = new List<ModelInput>
            {
                new() { X = 1,  Y = 5.0f },
                new() { X = 2,  Y = 7.0f },
                new() { X = 3,  Y = 9.0f },
                new() { X = 4,  Y = 11.0f },
                new() { X = 5,  Y = 13.0f },
                new() { X = 6,  Y = 15.0f },
                new() { X = 7,  Y = 17.0f },
                new() { X = 8,  Y = 19.0f },
                new() { X = 9,  Y = 21.0f },
                new() { X = 10, Y = 23.0f },
                new() { X = 11, Y = 25.0f },
                new() { X = 12, Y = 27.0f }
            };
            // Convert the in-memory list into an IDataView, the data pipeline's input format.
            var dataView = mlContext.Data.LoadFromEnumerable(trainingData);


            // Build training pipeline
            // Steps:
            // 1) Copy the Y column to the expected label column name "Label".
            // 2) Concatenate feature columns into a single "Features" vector (only X here).
            // 3) Normalize features (mean-variance) to help gradient-based training converge.
            // 4) Use OnlineGradientDescent regression trainer (a linear model trained with SGD).
             var pipeline =
    mlContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: nameof(ModelInput.Y))
    .Append(mlContext.Transforms.Concatenate("Features", nameof(ModelInput.X)))
    .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
    .Append(mlContext.Regression.Trainers.OnlineGradientDescent(
        labelColumnName: "Label",
        featureColumnName: "Features",
        numberOfIterations: 500,
        learningRate: 0.1f));


            // Cross-validation
            // We run 3-fold cross-validation to estimate how well the model generalizes.
           var cvResults = mlContext.Regression.CrossValidate(
                data: dataView,
                estimator: pipeline,
                numberOfFolds: 3,
                labelColumnName: "Label");
                
            // Print per-fold metrics so you can see variance between folds.
            Console.WriteLine("=== Per-Fold Metrics ===");
            int fold = 1;
            foreach (var r in cvResults)
            {
                Console.WriteLine($"Fold {fold++}: R²={r.Metrics.RSquared:0.###}, RMSE={r.Metrics.RootMeanSquaredError:0.###}");
            }

            // Print averaged cross-validated metrics (summary of model performance).
            Console.WriteLine("\n=== Cross-Validated Metrics ===");
            Console.WriteLine($"Avg R² Score: {cvResults.Average(r => r.Metrics.RSquared):0.###}");
            Console.WriteLine($"Avg RMSE: {cvResults.Average(r => r.Metrics.RootMeanSquaredError):0.###}");

            // Train a final model on all available data so it benefits from every example.
            // Note: since we've already used cross-validation to estimate performance,
            // retraining on all data is a common step before making production predictions.
            var model = pipeline.Fit(dataView);

            // Create a prediction engine for single predictions (convenient for demos).
            var engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var prediction = engine.Predict(new ModelInput { X = 10 });

            Console.WriteLine();
            Console.WriteLine($"Prediction for X = 10 → Y ≈ {prediction.Prediction:0.###}");
        }
    }
}
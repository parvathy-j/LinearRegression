using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace LinearRegression
{
    // Input data structure
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
            var mlContext = new MLContext(seed: 1);

            // Sample training data
            var trainingData = new List<ModelInput>
            {
                new() { X = 1, Y = 5 },
                new() { X = 2, Y = 7 },
                new() { X = 3, Y = 9 },
                new() { X = 4, Y = 11 },
                new() { X = 5, Y = 13 }
            };

            var dataView = mlContext.Data.LoadFromEnumerable(trainingData);

            // Split data for evaluation
            var splitData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);

            // Build training pipeline
            var pipeline = mlContext.Transforms
                .Concatenate("Features", nameof(ModelInput.X))
                .Append(mlContext.Regression.Trainers.Sdca(
                    labelColumnName: nameof(ModelInput.Y),
                    featureColumnName: "Features"));

            // Train model
            var model = pipeline.Fit(splitData.TrainSet);

            // Evaluate model
            var predictions = model.Transform(splitData.TestSet);
            var metrics = mlContext.Regression.Evaluate(
                predictions,
                labelColumnName: nameof(ModelInput.Y));

            Console.WriteLine("=== Model Metrics ===");
            Console.WriteLine($"R² Score: {metrics.RSquared:0.###}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:0.###}");

            // Make prediction
            var engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            var input = new ModelInput { X = 10 };
            var result = engine.Predict(input);

            Console.WriteLine();
            Console.WriteLine($"Prediction for X = 10 → Y ≈ {result.Prediction:0.###}");
        }
    }
}

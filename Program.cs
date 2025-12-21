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

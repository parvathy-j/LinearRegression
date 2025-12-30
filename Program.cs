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

    // Model input/output types moved to the Core library
    using LinearRegression.Core; 

    class Program
    {
        static void Main()
        {
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

            var trainer = new ModelTrainer();
            var (avgR2, avgRmse, perFold) = trainer.Train(trainingData);

            // Print per-fold metrics so you can see variance between folds.
            Console.WriteLine("=== Per-Fold Metrics ===");
            int fold = 1;
            foreach (var m in perFold)
            {
                Console.WriteLine($"Fold {fold++}: R²={m.RSquared:0.###}, RMSE={m.RootMeanSquaredError:0.###}");
            }

            // Print averaged cross-validated metrics (summary of model performance).
            Console.WriteLine("\n=== Cross-Validated Metrics ===");
            Console.WriteLine($"Avg R² Score: {avgR2:0.###}");
            Console.WriteLine($"Avg RMSE: {avgRmse:0.###}");

            Console.WriteLine();
            Console.WriteLine($"Prediction for X = 10 → Y ≈ {trainer.Predict(10):0.###}");
        }
    }
}
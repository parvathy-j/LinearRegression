using System.Collections.Generic;
using System.IO;
using LinearRegression.Core;
using Xunit;

namespace LinearRegression.Tests;

public class ModelTrainerTests
{
    [Fact]
    public void Train_And_Predict_Produces_Expected_Result()
    {
        // Arrange: simple linear data y = 2x + 3
        var trainingData = new List<ModelInput>
        {
            new() { X = 1, Y = 5 },
            new() { X = 2, Y = 7 },
            new() { X = 3, Y = 9 },
            new() { X = 4, Y = 11 },
            new() { X = 5, Y = 13 }
        };

        var trainer = new ModelTrainer();

        // Act
        var (avgR2, avgRmse, perFold) = trainer.Train(trainingData);
        var prediction = trainer.Predict(10);

        // Assert: prediction should be near 23 (2*10+3)
        Assert.InRange(prediction, 22.0f, 24.0f);
        Assert.True(avgR2 > 0.9);
    }

    [Fact]
    public void Save_And_Load_Model_Roundtrip_Works()
    {
        var trainingData = new List<ModelInput>
        {
            new() { X = 1, Y = 5 },
            new() { X = 2, Y = 7 },
            new() { X = 3, Y = 9 }
        };

        var trainer = new ModelTrainer();
        trainer.Train(trainingData);

        var tmp = Path.GetTempFileName();
        try
        {
            trainer.Save(tmp);

            var loader = new ModelTrainer();
            loader.Load(tmp);

            var p = loader.Predict(4);
            Assert.InRange(p, 10.0f, 12.0f);
        }
        finally
        {
            File.Delete(tmp);
        }
    }
}

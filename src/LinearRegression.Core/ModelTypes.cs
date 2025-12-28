using Microsoft.ML.Data;

namespace LinearRegression.Core
{
    // Represents one row of input data
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
}

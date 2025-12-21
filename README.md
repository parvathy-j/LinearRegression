#  Linear Regression in C# (ML.NET + From Scratch)

This repository demonstrates my learning of **machine learning fundamentals** by implementing **linear regression in C#** using two approaches:

1. **ML.NET** – a full machine learning pipeline with training, evaluation, and prediction  
2. **From-scratch implementation** – computing the regression parameters manually using mathematics

The goal of this project is to understand both **how machine learning frameworks work** and **what is happening under the hood mathematically**.

---

##  Project Overview

Linear regression is one of the foundational algorithms in machine learning.  
In this project, I:

- Built a regression model using **ML.NET**
- Evaluated it using **k-fold cross-validation**
- Improved stability using **feature normalization** and **gradient descent**
- Implemented the same model **from scratch** using the closed-form least squares solution
- Compared both approaches to reinforce understanding

---

##  What I Learned

Through this project, I learned how to implement and evaluate a linear regression model in C# using ML.NET, while also understanding the underlying machine learning concepts rather than treating the framework as a black box.

Key learnings include:

- How to structure a machine learning pipeline in ML.NET, including feature preparation, normalization, training, and prediction.
- Why simple train/test splits can produce unreliable or undefined metrics (such as NaN R²) when working with small datasets.
- How k-fold cross-validation provides more stable and reliable evaluation metrics by testing the model across multiple data splits.
- The importance of feature normalization when using gradient-based optimization methods.
- How different regression trainers behave differently on small datasets, and how trainer choice impacts model stability and accuracy.
- How to interpret regression metrics such as R² and RMSE in practical terms.
- How linear regression works mathematically using the least squares method.

This project helped bridge the gap between theoretical machine learning concepts and their practical implementation in real code.

---

##  Tech Stack

- **C#**
- **.NET 8**
- **ML.NET**
- Cross-platform (Windows / macOS / Linux)

---

##  Dataset

The model is trained on a simple synthetic dataset following the relationship:
y = 2x + 3

This clean dataset makes it easy to verify that the model is learning correctly and producing sensible predictions.

---

## Repository Structure
.
├── LinearRegression/ # ML.NET implementation
│ └── Program.cs
│
├── FromScratch/ # Manual linear regression (math-based)
│ └── Program.cs
│
└── README.md

---

## ML.NET vs From-Scratch Comparison

### ML.NET Implementation
- Uses a full ML pipeline
- Feature normalization
- Online Gradient Descent trainer
- 3-fold cross-validation for reliable metrics
- Produces R² and RMSE for evaluation

### From-Scratch Implementation
- Uses the closed-form least squares solution
- Manually computes slope and intercept
- No ML libraries involved
- Helps validate and understand the ML.NET results

Comparing both implementations reinforced how modern ML frameworks abstract and optimise core mathematical operations.

---

##  How to Run

### Prerequisites
- .NET 8 SDK installed

Check your version:
```bash
dotnet --version
```

Run ML.NET version

cd LinearRegression
dotnet run

Run from-scratch version

cd FromScratch
dotnet run


Example Output (ML.NET)

=== Per-Fold Metrics ===
Fold 1: R²=0.993, RMSE=0.596
Fold 2: R²=0.953, RMSE=0.435
Fold 3: R²=0.976, RMSE=1.124

=== Cross-Validated Metrics ===
Avg R² Score: 0.974
Avg RMSE: 0.718

Prediction for X = 10 → Y ≈ 22.6

---





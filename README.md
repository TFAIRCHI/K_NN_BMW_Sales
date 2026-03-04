# KNN BMW Sales Model Classification

This project uses a K-Nearest Neighbors (KNN) classifier in Jupyter to predict **BMW model type** from sales and market indicators.

## Project Files
- `neighbor_bmw_sales.ipynb`: main notebook (data prep, model training, tuning, and evaluation)
- `bmw_global_sales_2018_2025.csv`: dataset used by the notebook

## Objective
Given numeric sales and macro features, classify each row into one of 8 BMW model classes:
`3 Series`, `5 Series`, `MINI`, `X3`, `X5`, `X7`, `i4`, `iX`.

## Dataset Summary
- Rows: `3072`
- Columns: `11`
- Target: `Model`
- Features used by notebook:
  - `Units_Sold`
  - `Avg_Price_EUR`
  - `Revenue_EUR`
  - `BEV_Share`
  - `Premium_Share`
  - `GDP_Growth`
  - `Fuel_Price_Index`
- Excluded columns: `Year`, `Month`, `Region`

## Notebook Workflow
1. Import libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
2. Load the CSV file.
3. Split feature matrix `X` and target `y`.
4. Perform train/test split (`test_size=0.4`, `random_state=42`, stratified by class).
5. Build pipeline:
   - `StandardScaler`
   - `KNeighborsClassifier`
6. Tune hyperparameters with `GridSearchCV` (5-fold CV):
   - `n_neighbors`: 1-20
   - `weights`: `uniform`, `distance`
   - `metric`: `euclidean`, `manhattan`
7. Evaluate best model using:
   - confusion matrix heatmap
   - one-vs-rest ROC curves with AUC per class

## Reproduced Results
Running the notebook logic on this dataset produced:
- Best params: `{'knn__metric': 'manhattan', 'knn__n_neighbors': 19, 'knn__weights': 'distance'}`
- Best CV accuracy: `0.6658`
- Test accuracy: `0.6452`

## Environment Setup
Use Python 3.10+ (or similar) and install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Run
From this folder:

```bash
jupyter notebook neighbor_bmw_sales.ipynb
```

Then run all cells in order.

## Notes
- The notebook currently contains only code cells (no markdown annotations).
- If you add new features or change split/tuning settings, expect accuracy and AUC values to change.

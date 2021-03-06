## Model Metrics Report
### ('exo_SVC', SVC(C=100, gamma=25))
#### Classification report
|    |   precision |   recall |   f1-score |
|---:|------------:|---------:|-----------:|
|  1 |    0.889764 | 0.965812 |   0.92623  |
|  2 |    0.85489  | 0.844237 |   0.84953  |
|  3 |    0.764151 | 0.613636 |   0.680672 |
#### Confusion Matrix
|    |   0 |   1 |   2 |
|---:|----:|----:|----:|
|  0 | 339 |   9 |   3 |
|  1 |  28 | 271 |  22 |
|  2 |  14 |  37 |  81 |
#### Matthews Correlation Coefficient
MCC:0.7727738157051403
### ('exo_randforest', RandomForestClassifier(max_depth=18, max_features=3, n_estimators=700))
#### Classification report
|    |   precision |   recall |   f1-score |
|---:|------------:|---------:|-----------:|
|  1 |    0.920548 | 0.957265 |   0.938547 |
|  2 |    0.853293 | 0.88785  |   0.870229 |
|  3 |    0.8      | 0.636364 |   0.708861 |
#### Confusion Matrix
|    |   0 |   1 |   2 |
|---:|----:|----:|----:|
|  0 | 336 |  11 |   4 |
|  1 |  19 | 285 |  17 |
|  2 |  10 |  38 |  84 |
#### Matthews Correlation Coefficient
MCC:0.8008172668263573
### ('exo_xgb', XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.1, max_delta_step=0, max_depth=2,
              min_child_weight=1, missing=nan, monotone_constraints='()',
              n_estimators=1000, n_jobs=8, num_parallel_tree=1,
              objective='multi:softprob', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None))
#### Classification report
|    |   precision |   recall |   f1-score |
|---:|------------:|---------:|-----------:|
|  1 |    0.94382  | 0.957265 |   0.950495 |
|  2 |    0.867868 | 0.900312 |   0.883792 |
|  3 |    0.773913 | 0.674242 |   0.720648 |
#### Confusion Matrix
|    |   0 |   1 |   2 |
|---:|----:|----:|----:|
|  0 | 336 |   9 |   6 |
|  1 |  12 | 289 |  20 |
|  2 |   8 |  35 |  89 |
#### Matthews Correlation Coefficient
MCC:0.8193252854415962

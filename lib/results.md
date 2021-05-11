## Logistic Regression
precision    recall  f1-score   support

0       0.38      0.39      0.38        38
1       0.33      0.29      0.31        14
2       0.00      0.00      0.00         9
3       0.39      0.45      0.42        29
4       0.11      0.10      0.11        10
5       0.26      0.29      0.27        21
6       0.26      0.25      0.26        20

micro avg       0.31      0.31      0.31       141
macro avg       0.25      0.25      0.25       141
weighted avg       0.30      0.31      0.30       141
samples avg       0.29      0.37      0.29       141






## LSTM, 0.05 alpha, 10 epochs, no Glove, 2 outputs:
[[40 32]
 [ 8 10]]
              precision    recall  f1-score   support

           0       0.83      0.56      0.67        72
           1       0.24      0.56      0.33        18

    accuracy                           0.56        90
   macro avg       0.54      0.56      0.50        90
weighted avg       0.71      0.56      0.60        90

## LSTM, 0.05 alpha, 10 epochs, with Glove, 2 outputs:
[[29 43]
 [ 5 13]]
              precision    recall  f1-score   support

           0       0.85      0.40      0.55        72
           1       0.23      0.72      0.35        18

    accuracy                           0.47        90
   macro avg       0.54      0.56      0.45        90
weighted avg       0.73      0.47      0.51        90

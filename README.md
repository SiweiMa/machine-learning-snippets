# machine learning code snippets

## explain_binary_results.py

### Usage

```python
y_test = np.array([0, 0, 0, 1, 1, 1, 1, 0])
y_pred = np.array([0, 1, 1, 1, 0, 1, 0, 1])

explain_binary_classification(y_pred, y_test, 'fraud') # 'fraud' represents positive/1 case
```

### Output

```
                 Predicted      
                 not fraud fraud
Actual not fraud         1     3
       fraud             2     2

* False positive rate is 75.00%: among all 4 cases of actually not fraud, there are 3 cases wrongly predicted as fraud

* True positive rate (also called recall) is 50.00%: among all 4 cases of actually fraud, there are 2 cases correctly predicted as fraud

* Precision is 40.00%: among all 5 cases predicted as fraud, there are 2 cases correctly predicted as fraud

* Accuracy is 37.50%: among all 8 cases, there are 3 cases have been correctly predicted
```



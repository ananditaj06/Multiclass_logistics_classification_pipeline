# Multiclass Logistic Basics

Write a function to compute the **multiclass** logistic loss (also called the cross-entropy loss) given the parameters *(W, b)* of a linear model (as numpy arrays) and an example *(x, y)*.

Add an L1 regularization and an L2 regularization to the loss function.

---

## Classification Pipeline

- Generate data from `Data_Linear_Classifier.ipynb`.
- Split the data into test and train (20%:80%).
- Build a linear classifier assuming the multiclass logistic loss and an L2 regularization for the weights only.  
  Report the prediction accuracy on the training data and the test data and show appropriate plots.
- Introduce a cross-validation scheme and justify your choice of parameters.  
  What is the validation accuracy compared to the test accuracy?
- Analyze the sensitivity of the model's performance to different learning rates and the number of gradient descent iterations.  
  Describe via suitable plots.
- Examine the sensitivity of the model's performance to different regularization parameter values.  
  Find the best regularization parameter using an exhaustive search procedure.  
  Describe your choice via suitable plots.  
  What is the performance difference between using regularization and no regularization?
- Investigate the sensitivity of the model's performance with respect to a different test-train split (e.g., 50%:50%).

---

### Requirements
Ensure you have the following libraries installed before running the code:
```bash
pip install numpy pandas matplotlib scikit-learn

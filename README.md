# PA3: Comparing Classifiers

**Author:** Justin Shapiro

**Date:** March 11th, 2018

**Class:** CSCI-5800 (Special Topics: Machine Learning)

### Description
This assignment implements from scratch three Machine Learning classifiers: Naïve Bayes and two versions of Logistic Regression, one using Batch Gradient Descent (BGD-LR) to find `w` and one using Stochastic Gradient Descent (SGD-LR) to find `w`. These three classifiers are used to classify face data from the MIT Center for Biological and Computational Learning (CBCL) lab. After using the classifiers to train three different classification models, the models are used to predict whether a given image is a face. The metrics relating to model performance are then compared for each classifier side-by-side. 

### System Information
- **Programming Language:** Python 3.6.4 (64-bit) (*not* using Anaconda)
- **Operating System:** Windows 10 Pro
- **Architecture:**: 64-bit, Intel Xeon CPU E3-1270 @ 3.80GHz

### Required Packages
- [scikit-learn 0.19.1](https://pypi.python.org/pypi/scipy/1.0.0) (for preprocessing and normalization)
- [SciPy 1.0.0](https://pypi.python.org/pypi/scipy/1.0.0) (required for scikit-learn to run)
- [NumPy 1.14.0](https://pypi.python.org/pypi/numpy/1.14.0) (for linear algebra)

### Compile & Run
Make sure you run a **64-bit version** of Python 3.3 or greater version at 64-bits. The program will not work with a
32-bit version of Python because of the vectorization used to speed-up computation. Because vectorization is used,
there is often more than 4GB of RAM required to carry out operations on large matrices. Analytical vectorization is used
to find `w`.

Once a 64-bit version of Python 3.3 or greater is install along with the required packages listed above, start the
program with the script:

`python face_predictor.py`

## Report

### Performance Metrics

| Classifier    | Accuracy           | Precision          | Recall             | F1-Score           |
|---------------|--------------------|--------------------|--------------------|--------------------|
| *Naive Bayes* | 0.875857766687461  | 0.9823805060918464 | 0.8893225300131506 | 0.9335381738026852 |
| *BGD-LR*      | 0.8571012684549802 | 0.9827859026612323 | 0.8694693081067323 | 0.9226613847123436 |
| *SGD-LR*      | 0.9792056560615513 | 0.9803872579637727 | 0.9987697789844313 | 0.9894931495334959 |

### ROC Plots

![Naive Bayes ROC Curve](images/roc_1.png)

![BGD-LR ROC Curve](images/roc_2.png)

![SGD-LR ROC Curve](images/roc_3.png)

### Comparing Classifiers: Performance

In this assignment, we compared the performance differences between generative and discriminative classifiers on a facial recognition dataset. The generative classifier (Naïve Bayes classifier) is compared against the two discriminative classifiers (BGD-LR and SGD-LR), but the two discriminative classifiers are also compared up against each other. Surprisingly, we did not see a major performance difference between these two groups of classifiers alone. The performance differences are apparent when you compare them side-by-side rather than by category. The Naïve Bayes classifier performed slightly better than the BGD-LR classifier, but the SGD-LR classifier performed significantly better than the rest of the group. 
Accuracy is perhaps the best way to compare these classifiers in this case. The SGD-LR classifier (“pure” at a batch size of 1) had a near-perfect accuracy at almost 98%, getting slightly better as batch sizes increased. The Naïve Bayes and BGD-LR classifiers had an 87% and 85% prediction accuracy, respectively. While the accuracy score between the classifiers are different, they all have almost the exact same ratio of correct positive observations (known as precision). The differences between each classifier’s recall and F1-score are just as distributed as their respective accuracy scores, so they don’t provide any further insight. The ROC curves for each of the classifiers further back up the notion that the SGD-LR classifier is the most accurate classifier, with Naïve Bayes and BGD-LR competing.

While SGD-LR is perhaps the most accurate classifier in this scenario, that does not come without some real-world tradeoffs. Consider the following table of training and prediction times for all three classifiers:

| Classifier    | Training Time (s)  | Prediction Time (s) | 
|---------------|--------------------|---------------------|
| *Naive Bayes* | 0.011742           | 0.164202            | 
| *BGD-LR*      | 37.579959          | 3.98595             | 
| *SGD-LR*      | 105.663727         | 4.038198            |

Although the SGD-LR classifier was instructed to run at 1000 epochs while the BGD-LR classifier was instructed to run at only 100 epochs, the SGD-LR classifier is indeed slower to train than all the other classifiers. So, deciding which classifier to use depends on the application that we are using the machine learning model for. If accuracy is the most important metric to consider, SGD-LR is by far the best classifier to use because we would not care how long it would take to train, and prediction times are reasonable for anything that does not require real-time predictions. However, if we seek to use these classifiers to perform facial recognition in real time (like mobile device applications frequently do), Naïve Bayes would become the best classifier. The training and prediction time of the Naïve Bayes classifier is extraordinary good, and its accuracy is acceptable at 87%. Because of that, it is well suited for use in real-time applications. This all being said, there does not seem to be any good reason to use BGD-LR over any other classifier in any kind of application. The BGD-LR classifier really has no advantages, as its accuracy is the worst over all classifiers and its 4 second prediction time is not suited for real-time applications.

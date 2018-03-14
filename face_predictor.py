import os
import numpy as np
from datetime import datetime as time
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as metrics


# Returns a model that perform Naive Bayes classification
# Input: training_data: used to fit the NB classifier
# Output: A model function that, when given an input, uses Naive Bayes classifier to predict which class it belongs in
def naive_bayes_model(training_data):
    # define our training variables
    X = training_data[0]
    y = training_data[1]

    # group our sets of attributes according to the class they belong in
    grouped_classes = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]

    # compute the prior probability, taking the log to prevent floating point underflow
    m = X.shape[0]
    prior_probability = [np.log(len(m_c) / m) for m_c in grouped_classes]

    # compute the conditional probabilities of the features, using Laplace smoothing, and again taking the log
    m_ic = np.array([np.array(m_c).sum(axis=0) for m_c in grouped_classes])
    m_c = m_ic.sum(axis=1)[np.newaxis].T
    k = 256  # this is the number of of possible values a feature can have, which in this case is |{0, ..., 255}| = 256
    feature_probability = np.log((m_ic + 1) / (m_c + k))

    def nb_predictor(X_input):
        return np.argmax([prior_probability + (feature_probability * x).sum(axis=1) for x in X_input], axis=1)

    return nb_predictor


# Logistic regression function (BGD and SGD capable) which produces a model (function) and parameter w
# Input: training_data: used to train a model, refining w
#        num_epochs: the number of times we run gradient descent before settling for a value of w
#        learning_rate: the learning rate for the model
#        regularization_parameter: the lambda value used during regularization in gradient descent
#        batch: specifies whether we use batch or stochastic gradient descent
#        stochastic: specifies whether we use stochastic gradient descent
#        batch_size: the mini-batch size used in stochastic gradient descent, only used in stochastic=True
#        verbose: print verbose output during model training
# Output: h(x, w): the logistic regression hypothesis function
#         w: the value of w that we obtained from the training data, in other words: this is the model parameter
def logistic_regression_model(training_data,
                              num_epochs,
                              learning_rate=0.5,
                              regularization_parameter=0.5,
                              batch=False,
                              stochastic=False,
                              batch_size=1,  # purely SGD at 1, for values > 1, it's mini-batch gradient descent
                              verbose=False):
    if batch == stochastic:
        if batch and stochastic:
            raise AttributeError("Either Batch or Stochastic Gradient Descent can be performed, but not both")
        else:
            raise AttributeError("You must choose either Batch or Stochastic Gradient Descent")

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def h(x, w):
        # this prediction function should generalize to batch and single predictions
        if x.shape[0] != w.shape[0]:
            w = np.tile(w, (x.shape[0], 1))

        # obtaining thousands of predictions with a for loop takes HOURS on CPU
        # this is a vectorized implementation of batch prediction that completes in only seconds
        # swap the operands before multiply to ensure the prediction results exit in the resultant matrix's diagonal
        # although the operands are swapped, this will also work for a single prediction
        return sigmoid(np.diagonal(np.matmul(x, np.transpose(w))))  # must have 64-bit Python for this to work

    # perform Gradient Descent to find w
    def gradient_descent(X, y):
        w = np.random.uniform(size=(X.shape[1],))

        if batch:
            for epoch in np.arange(0, num_epochs):
                error = h(X, w) - y
                w -= (learning_rate * X.T.dot(error)) - (regularization_parameter * w)

                if verbose:
                    print("Loss: " + str(np.sum(error ** 2)) + ", Epoch: " + str(epoch))

        elif stochastic:
            def next_batch():
                for i in np.arange(0, X.shape[0]):
                    yield (X[i: i + batch_size], y[i: i + batch_size])

            w = np.random.uniform(size=(X.shape[1],))
            for epoch in np.arange(0, num_epochs):
                losses = []
                for (X_batch, y_batch) in next_batch():
                    error = h(X_batch, w) - y_batch
                    w -= (learning_rate * X_batch.T.dot(error))

                    if verbose:
                        losses.append(np.sum(error ** 2))

                if verbose:
                    print("Loss: " + str(np.average(losses)) + ", Epoch: " + str(epoch))

        return w

    X_train = training_data[0]
    y_train = training_data[1]

    # return the model function and model parameter
    return h, gradient_descent(X_train, y_train)


# Read binary data from PGM image files to a n-dimensional vector
def pgm_to_face_vector(file, is_face):
    return np.append(
        # add bias
        [1],
        np.append(
            # We already know that the images are 19x19 pixes, so the count will be 361 and the offset will be 13
            # The data type will be "u1" since the image is greyscale (meaning each element does not exceed 255)
            np.frombuffer(open(file, 'rb').read(), dtype='u1', count=361, offset=13).reshape(361),

            # add true/false label
            [1 if is_face else 0]
        )
    )


# Read data into a 361 dimensional vector
this_location = str(os.path.dirname(os.path.realpath(__file__)))
FACE_TRAIN_DIR = this_location + '\\MIT-CBCL-Face-dataset\\train\\face'
NON_FACE_TRAIN_DIR = this_location + '\\MIT-CBCL-Face-dataset\\train\\non-face'
FACE_TEST_DIR = this_location + '\\MIT-CBCL-Face-dataset\\test\\face'
NON_FACE_TEST_DIR = this_location + '\\MIT-CBCL-Face-dataset\\test\\non-face'
FACE_TRAIN_DATA = [pgm_to_face_vector(FACE_TRAIN_DIR + "\\" + file, True) for file in os.listdir(FACE_TRAIN_DIR)]
NON_FACE_TRAIN_DATA = [pgm_to_face_vector(NON_FACE_TRAIN_DIR + "\\" + file, False) for file in os.listdir(NON_FACE_TRAIN_DIR)]
FACE_TEST_DATA = [pgm_to_face_vector(FACE_TEST_DIR + "\\" + file, True) for file in os.listdir(FACE_TEST_DIR)]
NON_FACE_TEST_DATA = [pgm_to_face_vector(NON_FACE_TEST_DIR + "\\" + file, False) for file in os.listdir(NON_FACE_TEST_DIR)]

# Assert that the data is of the correct shape
# There are 361 features per sample, so '363' includes the bias term and the target/label
assert(np.shape(FACE_TRAIN_DATA) == (2429, 363))
assert(np.shape(NON_FACE_TRAIN_DATA) == (4548, 363))
assert(np.shape(FACE_TEST_DATA) == (472, 363))
assert(np.shape(NON_FACE_TEST_DATA) == (23573, 363))

# Group the training and test data into single train/test sets (convert to NumPy array while we're at it)
TRAIN_DATA = FACE_TRAIN_DATA + NON_FACE_TRAIN_DATA
TEST_DATA = FACE_TEST_DATA + NON_FACE_TEST_DATA

# Separate the training/testing set into X and y sets
X_train = np.delete(TRAIN_DATA, -1, axis=1)
y_train = np.reshape([row[-1] for row in TRAIN_DATA], (len(TRAIN_DATA),))
X_test = np.delete(TEST_DATA, -1, axis=1)
y_test = np.reshape([row[-1] for row in TEST_DATA], (len(TEST_DATA),))

m_train, n_train = X_train.shape
m_test, n_test = X_test.shape

# We expect the number of samples in the training set to be 2429 + 4548 = 6977
# We also expect there to be 2429 "face" (1) targets and 4548 "not face" (0) targets
assert(m_train == 6977 and len(FACE_TRAIN_DATA) == 2429 and len(NON_FACE_TRAIN_DATA) == 4548)

# We expect the number of samples in the testing dataset to be 472 + 23573 = 24045
# We also expect there to be 472 "face" (1) targets and 23573 "not face" (0) targets
assert(m_test == 24045 and len(FACE_TEST_DATA) == 472 and len(NON_FACE_TEST_DATA) == 23573)

# Now that the train/test sets are asserted to be correct, print a summary of the datasets we are about to work with
print("The training set has " + str(m_train) + " samples with " + str(n_train - 1) + " attributes per sample")
print("     -> There are " + str(len(FACE_TRAIN_DATA)) + " \"face\" (1) targets and " + str(len(NON_FACE_TRAIN_DATA)) + " \"not face\" (0) targets")
print("The testing set has " + str(m_test) + " samples with " + str(n_test - 1) + " attributes per sample")
print("     -> There are " + str(len(FACE_TEST_DATA)) + " \"face\" (1) targets and " + str(len(NON_FACE_TEST_DATA)) + " \"not face\" (0) targets")

##################
# Problems 1 - 3 #
##################

print("\nProblem #1 (Naive Bayes): \n------------------")
# Naive Bayes is a probability-based method and does not use the linear hypothesis,
# so we do not need to include the x_0 bias term
X_train_no_bias = np.delete(X_train, 0, axis=1)
X_test_no_bias = np.delete(X_test, 0, axis=1)

nb_train_start = time.now()
nb_model = naive_bayes_model(training_data=[X_train_no_bias, y_train])
nb_train_time = time.now() - nb_train_start

nb_pred_start = time.now()
nb_pred = nb_model(X_test_no_bias)
nb_pred_time = time.now() - nb_pred_start

try:
    nb_accuracy = accuracy_score(y_test, nb_pred)
except ValueError:
    nb_accuracy = accuracy_score(y_test, nb_pred.round())
    pass

try:
    nb_metrics = metrics(y_test, nb_pred)
except ValueError:
    nb_metrics = metrics(y_test, nb_pred.round())
    pass


nb_precision = nb_metrics[0][0]
nb_recall = nb_metrics[1][0]
nb_f1_score = nb_metrics[2][0]
nb_false_positive_rates, nb_true_positive_rates, nb_thresholds = roc_curve(y_test, nb_pred)
print("Naive Bayes Accuracy: " + str(nb_accuracy))
print("Naive Bayes Precision: " + str(nb_precision))
print("Naive Bayes Recall: " + str(nb_recall))
print("Naive Bayes F1-Score: " + str(nb_f1_score))
print("Naive Bayes ROC Curve: ")
print("    -> False Positive Rates: " + str(nb_false_positive_rates))
print("    -> True Positive Rates: " + str(nb_true_positive_rates))
print("    -> Thresholds: " + str(nb_thresholds))
print("Naive Bayes Training Time: " + str(nb_train_time.total_seconds()))
print("Naive Bayes Prediction Time: " + str(nb_pred_time.total_seconds()))

print("\nProblem #2 (BGD-LR): \n------------------")
bgd_lr_train_start = time.now()
h_batch, w_batch = logistic_regression_model(
    training_data=[X_train, y_train],
    num_epochs=100,
    batch=True
)
bgd_lr_train_time = time.now() - bgd_lr_train_start

bgd_lr_pred_start = time.now()
y_bgd_lr_pred = h_batch(X_test, w_batch)
bgd_lr_pred_time = time.now() - bgd_lr_pred_start

try:
    bgd_lr_accuracy = accuracy_score(y_test, y_bgd_lr_pred)
except ValueError:
    bgd_lr_accuracy = accuracy_score(y_test, y_bgd_lr_pred.round())
    pass

try:
    bgd_lr_metrics = metrics(y_test, y_bgd_lr_pred)
except ValueError:
    bgd_lr_metrics = metrics(y_test, y_bgd_lr_pred.round())
    pass

bgd_lr_precision = bgd_lr_metrics[0][0]
bgd_lr_recall = bgd_lr_metrics[1][0]
bgd_lr_f1_score = bgd_lr_metrics[2][0]
bgd_lr_false_positive_rates, bgd_lr_true_positive_rates, bgd_lr_thresholds = roc_curve(y_test, y_bgd_lr_pred, pos_label=2)

print("BGD-LR Accuracy: " + str(bgd_lr_accuracy))
print("BGD-LR Precision: " + str(bgd_lr_precision))
print("BGD-LR Recall: " + str(bgd_lr_recall))
print("BGD-LR F1-Score: " + str(bgd_lr_f1_score))
print("BGD-LR ROC Curve: ")
print("    -> False Positive Rates: " + str(bgd_lr_false_positive_rates))
print("    -> True Positive Rates: " + str(bgd_lr_true_positive_rates))
print("    -> Thresholds: " + str(bgd_lr_thresholds))
print("BGD-LR Training Time: " + str(bgd_lr_train_time.total_seconds()))
print("BGD-LR Prediction Time: " + str(bgd_lr_pred_time.total_seconds()))

print("\nProblem #3 (SGD-LR): \n------------------")
sgd_lr_train_start = time.now()
h_stochastic, w_stochastic = logistic_regression_model(
    training_data=[X_train, y_train],
    num_epochs=1000,
    stochastic=True
)
sgd_lr_train_time = time.now() - sgd_lr_train_start

sgd_lr_pred_start = time.now()
y_sgd_lr_pred = h_stochastic(X_test, w_stochastic)
sgd_lr_pred_time = time.now() - sgd_lr_pred_start

try:
    sgd_lr_accuracy = accuracy_score(y_test, y_sgd_lr_pred)
except ValueError:
    sgd_lr_accuracy = accuracy_score(y_test, y_sgd_lr_pred.round())
    pass

try:
    sgd_lr_metrics = metrics(y_test, y_sgd_lr_pred)
except ValueError:
    sgd_lr_metrics = metrics(y_test, y_sgd_lr_pred.round())
    pass

sgd_lr_precision = sgd_lr_metrics[0][0]
sgd_lr_recall = sgd_lr_metrics[1][0]
sgd_lr_f1_score = sgd_lr_metrics[2][0]
sgd_lr_false_positive_rates, sgd_lr_true_positive_rates, sgd_lr_thresholds = roc_curve(y_test, y_sgd_lr_pred)

print("SGD-LR Accuracy: " + str(sgd_lr_accuracy))
print("SGD-LR Precision: " + str(sgd_lr_precision))
print("SGD-LR Recall: " + str(sgd_lr_recall))
print("SGD-LR F1-Score: " + str(sgd_lr_f1_score))
print("SGD-LR ROC Curve: ")
print("    -> False Positive Rates: " + str(sgd_lr_false_positive_rates))
print("    -> True Positive Rates: " + str(sgd_lr_true_positive_rates))
print("    -> Thresholds: " + str(sgd_lr_thresholds))
print("SGD-LR Training Time: " + str(sgd_lr_train_time.total_seconds()))
print("SGD-LR Prediction Time: " + str(sgd_lr_pred_time.total_seconds()))

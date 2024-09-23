import numpy as np
import matplotlib.pyplot as plt

data = np.load('C:\\Users\\manis\\Downloads\\mnist.npz')
list(data.keys())

x_train = data['x_train']
y_train = data['y_train']
x_test = data['x_test']
y_test = data['y_test']

train_filter = np.where((y_train == 0) | (y_train == 1))
test_filter = np.where((y_test == 0) | (y_test == 1))

x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

y_train_filtered = np.where(y_train_filtered == 0, -1, 1)
y_test_filtered = np.where(y_test_filtered == 0, -1, 1)

val_indices_1 = np.where(y_train_filtered == 1)[0][:1000]
val_indices_neg1 = np.where(y_train_filtered == -1)[0][:1000]

val_indices = np.concatenate([val_indices_1, val_indices_neg1])
train_indices = np.setdiff1d(np.arange(len(y_train_filtered)), val_indices)

x_val = x_train_filtered[val_indices]
y_val = y_train_filtered[val_indices]
x_train_final = x_train_filtered[train_indices]
y_train_final = y_train_filtered[train_indices]

print("----")
print(x_train_final.shape, y_train_final.shape, x_val.shape, y_val.shape, x_test_filtered.shape, y_test_filtered.shape)
print("----")

def pca(X, num_components):
    X_centered = X - np.mean(X, axis=0)

    covariance_matrix = np.cov(X_centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    components = sorted_eigenvectors[:, :num_components]

    X_reduced = np.dot(X_centered, components)

    return X_reduced, components


x_train_flat = x_train_final.reshape(x_train_final.shape[0], -1)
x_val_flat = x_val.reshape(x_val.shape[0], -1)
x_test_flat = x_test_filtered.reshape(x_test_filtered.shape[0], -1)

x_train_pca, pca_components = pca(x_train_flat, 5)
x_val_pca = np.dot(x_val_flat - np.mean(x_train_flat, axis=0), pca_components)
x_test_pca = np.dot(x_test_flat - np.mean(x_train_flat, axis=0), pca_components)

print(x_train_pca.shape, x_val_pca.shape, x_test_pca.shape)

#Q1

# def decision_stump_weighted(X, y, weights):
#     n_samples, n_features = X.shape
#     best_stump = {
#         'feature': None,
#         'threshold': None,
#         'polarity': 1,
#         'error': float('inf')
#     }
#
#     for feature_index in range(n_features):
#         feature_values = np.unique(X[:, feature_index])
#         thresholds = (feature_values[:-1] + feature_values[1:]) / 2
#
#         for threshold in thresholds:
#             for polarity in [1, -1]:
#                 predictions = np.where(X[:, feature_index] < threshold, -1, 1) * polarity
#                 misclassified = predictions != y
#                 weighted_error = np.dot(weights, misclassified)
#
#                 if weighted_error < best_stump['error']:
#                     best_stump = {
#                         'feature': feature_index,
#                         'threshold': threshold,
#                         'polarity': polarity,
#                         'error': weighted_error
#                     }
#
#     return best_stump


def decision_stump_weighted(X, y, weights, num_features_to_consider=None):
    n_samples, n_features = X.shape

    if num_features_to_consider is None:
        num_features_to_consider = n_features

    considered_features = np.random.choice(n_features, num_features_to_consider, replace=False)

    best_stump = {
        'feature': None,
        'threshold': None,
        'polarity': 1,
        'error': float('inf')
    }

    for feature_index in considered_features:
        feature_values = np.unique(X[:, feature_index])
        thresholds = (feature_values[:-1] + feature_values[1:]) / 2

        if len(thresholds) > 0:
            threshold = np.random.choice(thresholds)

            for polarity in [1, -1]:
                predictions = np.where(X[:, feature_index] < threshold, -1, 1) * polarity
                misclassified = predictions != y
                weighted_error = np.dot(weights, misclassified)

                if weighted_error < best_stump['error']:
                    best_stump = {
                        'feature': feature_index,
                        'threshold': threshold,
                        'polarity': polarity,
                        'error': weighted_error
                    }

    return best_stump


def ada_boost_with_validation(X_train, y_train, X_val, y_val, T, num_features_to_consider):
    n_samples, n_features = X_train.shape
    weights = np.full(n_samples, 1 / n_samples)
    classifiers = []
    val_accuracies = []

    for t in range(T):
        print(t)
        stump = decision_stump_weighted(X_train, y_train, weights, num_features_to_consider)
        predictions = np.where(X_train[:, stump['feature']] < stump['threshold'], -1, 1) * stump['polarity']
        misclassified = predictions != y_train
        error = np.dot(weights, misclassified)

        alpha = 0.5 * np.log((1 - error) / error) if error != 0 else float('inf')
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)

        stump['alpha'] = alpha
        classifiers.append(stump)

        val_predictions = predict(X_val, classifiers)
        val_accuracy = np.mean(val_predictions == y_val)
        val_accuracies.append(val_accuracy)

    return classifiers, val_accuracies


def predict(X, classifiers):
    model_output = np.zeros(len(X))
    for classifier in classifiers:
        predictions = np.where(X[:, classifier['feature']] < classifier['threshold'], -1, 1) * classifier['polarity']
        model_output += classifier['alpha'] * predictions
    return np.sign(model_output)


T = 300

num_features_to_consider = 3

classifiers, val_accuracies = ada_boost_with_validation(x_train_pca, y_train_final, x_val_pca, y_val, T, num_features_to_consider)

print(classifiers, val_accuracies)

plt.figure(figsize=(10, 5))
plt.plot(range(1, T + 1), val_accuracies, marker='o', linestyle='-', color='b')
plt.title('Validation Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()



best_tree_index = np.argmax(val_accuracies)
print(f"Best tree count: {best_tree_index + 1}, Validation Accuracy: {val_accuracies[best_tree_index]}")

test_predictions = predict(x_test_pca, classifiers[:best_tree_index + 1])
test_accuracy = np.mean(test_predictions == y_test_filtered)
print(f"Test Set Accuracy: {test_accuracy}")

#------------------------------------------------------------------

#Q2
def mean_squared_error(actual, predicted):

    differences = np.array(actual) - np.array(predicted)
    squared_differences = differences ** 2
    mean_squared_error = squared_differences.mean()
    return mean_squared_error

# def fit_stump(X, y):
#     best_stump = {'feature': None, 'threshold': None, 'error': float('inf')}
#     for feature_index in range(X.shape[1]):
#         thresholds = np.unique(X[:, feature_index])
#         for i in range(len(thresholds) - 1):
#             threshold = (thresholds[i] + thresholds[i + 1]) / 2
#             for polarity in [1, -1]:
#                 predictions = np.where(X[:, feature_index] <= threshold, polarity, -polarity)
#                 error = np.sum((predictions - y) ** 2)
#                 if error < best_stump['error']:
#                     best_stump = {'feature': feature_index, 'threshold': threshold, 'error': error,
#                                   'polarity': polarity}
#     return best_stump



def fit_stump(X, y, num_features_to_consider=10, num_thresholds_to_consider=5):
    n_samples, n_features = X.shape
    best_stump = {'feature': None, 'threshold': None, 'error': float('inf')}

    # Randomly select a subset of features to consider
    if n_features > num_features_to_consider:
        features_indices = np.random.choice(n_features, num_features_to_consider, replace=False)
    else:
        features_indices = np.arange(n_features)

    for feature_index in features_indices:
        thresholds = np.unique(X[:, feature_index])

        # Randomly select thresholds to consider, if specified
        if len(thresholds) > num_thresholds_to_consider:
            threshold_indices = np.random.choice(len(thresholds) - 1, num_thresholds_to_consider, replace=False)
            selected_thresholds = [(thresholds[i] + thresholds[i + 1]) / 2 for i in sorted(threshold_indices)]
        else:
            selected_thresholds = [(thresholds[i] + thresholds[i + 1]) / 2 for i in range(len(thresholds) - 1)]

        for threshold in selected_thresholds:
            for polarity in [1, -1]:
                predictions = np.where(X[:, feature_index] <= threshold, polarity, -polarity)
                error = np.sum((predictions - y) ** 2)
                if error < best_stump['error']:
                    best_stump = {'feature': feature_index, 'threshold': threshold, 'error': error,
                                  'polarity': polarity}

    return best_stump


n_trees = 300
learning_rate = 0.01
models = []
residuals = y_train_final.astype(np.float64)
mse_scores = []

for i in range(n_trees):
    print(i)
    stump = fit_stump(x_train_pca, residuals, num_features_to_consider = 5, num_thresholds_to_consider = 10)
    predictions = np.where(x_train_pca[:, stump['feature']] <= stump['threshold'], stump['polarity'],
                           -stump['polarity'])
    residuals -= learning_rate * predictions
    models.append(stump)

    val_predictions = np.zeros_like(y_val, dtype=np.float64)
    for model in models:
        val_predictions += learning_rate * np.where(x_val_pca[:, model['feature']] <= model['threshold'],
                                                    model['polarity'], -model['polarity'])
    mse = mean_squared_error(y_val, val_predictions)
    mse_scores.append(mse)

plt.figure(figsize=(10, 5))
plt.plot(range(n_trees), mse_scores, marker='o', linestyle='-', color='b')
plt.title('Validation MSE vs. Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('MSE')
plt.grid(True)
plt.show()

best_model_index = np.argmin(mse_scores)
print(f"Best model at tree {best_model_index + 1} with MSE {mse_scores[best_model_index]}")

test_predictions = np.zeros_like(y_test_filtered, dtype=np.float64)
for model in models[:best_model_index + 1]:
    test_predictions += learning_rate * np.where(x_test_pca[:, model['feature']] <= model['threshold'],
                                                 model['polarity'], -model['polarity'])
final_mse = mean_squared_error(y_test_filtered, test_predictions)
print(f"Test MSE: {final_mse}")

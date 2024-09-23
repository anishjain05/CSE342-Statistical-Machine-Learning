import numpy as np

with np.load('C:\\Users\\manis\\Downloads\\mnist.npz', allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

train_filter = np.isin(y_train, [0, 1, 2])
test_filter = np.isin(y_test, [0, 1, 2])

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

print(f'Training set shape: {x_train.shape}')
print(f'Test set shape: {x_test.shape}')


# --------------------------------------------------------
# --------------------------------------------------------

def standardize_data(data):
    """ Standardize the data to have a mean of 0 and std deviation of 1. """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def pca(data, num_components):
    """ Perform PCA and reduce the data dimensions to num_components. """
    data_standardized = standardize_data(data)

    covariance_matrix = np.cov(data_standardized, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    data_reduced = np.dot(eigenvector_subset.transpose(), data_standardized.transpose()).transpose()

    return data_reduced


num_components = 10
x_train_pca = pca(x_train, num_components)
x_test_pca = pca(x_test, num_components)

print(f'Training set shape after PCA: {x_train_pca.shape}')
print(f'Test set shape after PCA: {x_test_pca.shape}')


# --------------------------------------------------------
# --------------------------------------------------------

def gini_index(groups, classes):
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            proportion = [row[-1] for row in group].count(class_val) / size
            score += proportion * proportion
        gini += (1.0 - score) * (size / n_instances)
    return gini


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def get_best_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_best_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_best_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


def build_tree(train, max_depth, min_size):
    root = get_best_split(train)
    split(root, max_depth, min_size, 1)
    return root


def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# ------------------------------------------------------------------

train_data = np.hstack((x_train_pca, y_train[:, None]))

test_data = np.hstack((x_test_pca, y_test[:, None]))

max_depth = 2  # For 3 terminal nodes as given in question
min_size = 1

tree = build_tree(train_data.tolist(), max_depth, min_size)


def predict_dataset(tree, dataset):
    predictions = list()
    for row in dataset:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


predictions = predict_dataset(tree, test_data[:, :-1].tolist())

actual = test_data[:, -1]
correct = sum(1 for i in range(len(actual)) if actual[i] == predictions[i])
accuracy = correct / float(len(actual)) * 100.0

print(f'Accuracy: {accuracy:.2f}%')


# -----------------------------------------------------------------

def create_bootstrapped_datasets(data, n_datasets):
    bootstrapped_datasets = []
    n_samples = len(data)
    for _ in range(n_datasets):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrapped_dataset = data[indices]
        bootstrapped_datasets.append(bootstrapped_dataset)
    return bootstrapped_datasets


def train_trees_on_bootstraps(bootstrapped_datasets, max_depth, min_size):
    trees = []
    for dataset in bootstrapped_datasets:
        tree = build_tree(dataset.tolist(), max_depth, min_size)
        trees.append(tree)
    return trees


def majority_voting(trees, sample):
    predictions = [predict(tree, sample) for tree in trees]
    return max(set(predictions), key=predictions.count)


def bagging_predict(trees, dataset):
    predictions = [majority_voting(trees, row.tolist()) for row in dataset]
    return predictions


n_trees = 5
bootstrapped_datasets = create_bootstrapped_datasets(train_data, n_trees)

trees = train_trees_on_bootstraps(bootstrapped_datasets, max_depth, min_size)  # 1 tree for each bootstrapped dataset

bagging_predictions = bagging_predict(trees, test_data[:, :-1])

correct_bagging = sum(1 for i in range(len(actual)) if actual[i] == bagging_predictions[i])
accuracy_bagging = correct_bagging / float(len(actual)) * 100.0

print(f'Bagging Accuracy: {accuracy_bagging:.2f}%')




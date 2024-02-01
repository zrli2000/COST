from typing import Counter
import pandas as pd
from sklearn.metrics import accuracy_score
from .fisherLib import fisher_exact as fisher_exact_log
from scipy.stats import fisher_exact as fisher_exact_scipy
from scipy import stats
from scipy.stats import chi2
import numpy as np


class COST_Classifier:
    # 初始化
    def __init__(self):
        """init fuction
        """
        # self.log_factorial = self.get_log_factorial()
        pass

    def fit(self, X, y, method):
        """get the information of features and labels

        Args: 
            method (str): the method to predict
            X (ndarry): the feature data
            y (array): the label data
        """
        self.information = {}
        self.train_X = X
        self.train_y = y
        self.feature_information(self.train_X, self.train_y, method)

    def get_log_factorial(self):
        """get the log value of factorial

        Returns:
            array: store the log value of factorial, such as log_factorial[5] = log(5!)
        """
        max_factorial = 15000
        log_factorial = np.zeros(max_factorial + 1)
        for i in range(1, max_factorial + 1):
            log_factorial[i] = log_factorial[i - 1] + np.log(i)
        return log_factorial

    def process(self, arr, labels):
        """get the information of one feature and it's information with labels

        Args:
            arr (array): one feature
            labels (array): label

        Returns:
            map: unique_value and it's count and it's label information
        """
        value_index_map = {}
        # visit every value
        for index, value in enumerate(arr):
            if value not in value_index_map:
                value_index_map[value] = {'count': 0, 'labels': []}
            value_index_map[value]['count'] += 1
            value_index_map[value]['labels'].append(labels[index])
        # Convert label lists to numpy arrays
        for value in value_index_map:
            value_index_map[value]['labels'] = np.array(value_index_map[value]['labels'])
        return value_index_map


    def feature_information(self, X, y, method):
        """get each feature's information so we can get it's p-value

        Args:
            X (_type_): _description_
            y (_type_): _description_
        """
        sample_num = X.shape[0]

        label_map = {}
        for label in y:
            if label in label_map:
                label_map[label] += 1
            else:
                label_map[label] = 1
        classes = sorted(set(y))
        for col in range(X.shape[1]):
            value_index_map = self.process(X[:, col], np.array(y))
            for value, info in value_index_map.items():
                a_b = info['count']
                label_counts = np.bincount(info['labels'])
                for t in classes: # in unique_labels
                    a = label_counts[t] if t < len(label_counts) else 0
                    b = a_b - a
                    c = label_map.get(t, 0) - a
                    d = sample_num - a - b - c
                    logp = self.ComputeFisher(np.array([[a, c], [b, d]]), method)
                    key = f"{t}-{col}-{value}"
                    self.information[key] = logp

    def ComputeFisher(self, table, method):
        """get p value by fisher's exact test

        Args:
            table (array of 2*2): 2*x contigency table
        
        Returns:
            float: p value(rOP) or log-p-value(fisher)
        """
        if method == 'sumP':
            return fisher_exact_log(table, alternative='greater')[1]
        elif method == 'fisher':
            return -2 * fisher_exact_log(table, alternative='greater')[1]
        else:
            return fisher_exact_scipy(table, alternative='greater')[1]

    def predict(self, train_X, train_y, valid_X, valid_y, test_X, unique_y_num, method):
        """according to the method to predict the test_X

        Args:
            train_X (ndarry): train_X
            train_y (array): train_y
            test_X (ndarry): test_X
            class_list (list): the list of class label
            method (str): the method to predict
            data_name (str): the name of dataset

        Returns:
            array: the predict result
        """
        class_list = list(range(unique_y_num))
        if method == 'minP':
            self.fit(np.concatenate([train_X, valid_X], axis=0), np.concatenate([train_y, valid_y]),method)
            res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p = self.predict_rOP(test_X, class_list, 0 )
            return res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p
        elif method == 'maxP':
            self.fit(np.concatenate([train_X, valid_X], axis=0), np.concatenate([train_y, valid_y]),method)
            res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p = self.predict_rOP(test_X, class_list, train_X.shape[1] - 1)
            return res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p
        elif method == 'rOP':
            r_best = self.checkR(train_X, train_y, valid_X, valid_y, class_list, method)
            res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p = self.predict_rOP(test_X, class_list, r_best)
            return res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p
        elif method == 'fisher':
            self.fit(np.concatenate([train_X, valid_X], axis=0), np.concatenate([train_y, valid_y]), method)
            res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p = self.predict_fisher(test_X, class_list)
            return res_regular, res_reject, res_regular_refine, res_reject_refine, r_table_p

    def predict_rOP(self, test_X, class_list, r):
        """the rOP predict method

        Args:
            test_X (ndarry): test_X
            class_list (list): the class label list
            r (int): the r generated by checkR()

        Returns:
            the rOP predict result
        """
        alpha = 0.05
        num_sample = test_X.shape[0]
        num_features = test_X.shape[1]
        info = np.ones((len(class_list), num_sample, num_features))
        for t in class_list:
            arr_t = np.ones((num_sample, num_features))
            for col in range(num_features):
                data_col = test_X[:, col]
                col_val, val_idx, counts = np.unique(data_col, return_inverse=True, return_counts=True)
                val_indices = np.split(np.argsort(val_idx), np.cumsum(counts[:-1]))
                for col_val_, val_indices_ in zip(col_val, val_indices):
                    key = f"{t}-{col}-{col_val_}"
                    arr_t[val_indices_, col] = self.information.get(key, 1)
            info[t] = np.sort(arr_t, axis=1)
        r_table = info[:, :, r].T
        r_table_p = stats.beta.cdf(r_table, r + 1, num_features - r)

        'Regular'
        res_regular = np.argmin(r_table_p, axis=1)

        'Reject'
        res_reject = np.argmin(r_table_p, axis=1)
        res_reject_p_value = np.min(r_table_p, axis=1)
        res_reject[res_reject_p_value >= alpha] = -1

        'regular & Refine'
        res_regular_refine = []
        for i, row in enumerate(r_table_p):
            significant = np.where(row < alpha)[0]
            if len(significant) == 1:
                res_regular_refine.append(significant[0])
            elif len(significant) > 1:
                res_regular_refine.append(significant)
            else:
                res_regular_refine.append(np.argmin(row))

        'Reject & Refine'
        res_reject_refine = []
        for i, row in enumerate(r_table_p):
            significant = np.where(row < alpha)[0]
            if len(significant) == 1:
                res_reject_refine.append(significant[0])
            elif len(significant) > 1:
                res_reject_refine.append(significant)
            else:
                res_reject_refine.append(-1)

        return res_regular, res_reject,res_regular_refine, res_reject_refine, r_table_p

    def predict_fisher(self, test_X, class_list):
        """the rOP predict method

        Args:
            test_X (ndarry): test_X
            class_list (list): the class label list

        Returns:
             the fisher predict result
        """
        alpha = 0.05
        num_sample = test_X.shape[0]
        num_features = test_X.shape[1]
        r_table_p = np.ones((num_sample, len(class_list)))
        for t in class_list:
            arr_t = np.zeros((num_sample, num_features))
            for col in range(num_features):
                data_col = test_X[:, col]
                col_val, val_idx, counts = np.unique(data_col, return_inverse=True, return_counts=True)
                val_indices = np.split(np.argsort(val_idx), np.cumsum(counts[:-1]))
                for col_val_, val_indices_ in zip(col_val, val_indices):
                    key = f"{t}-{col}-{col_val_}"
                    arr_t[val_indices_, col] = self.information.get(key, 0)
            for i in range(num_sample):
                r_table_p[i][t] = 1 - chi2.cdf(np.sum(arr_t[i,:]), num_features * 2)


        'Regular'
        res_regular = np.argmin(r_table_p, axis=1)

        'Reject'
        res_reject = np.argmin(r_table_p, axis=1)
        res_reject_p_value = np.min(r_table_p, axis=1)
        res_reject[res_reject_p_value >= alpha] = -1

        'regular & Refine'
        res_regular_refine = []
        for i, row in enumerate(r_table_p):
            significant = np.where(row < alpha)[0]
            if len(significant) == 1:
                res_regular_refine.append(significant[0])
            elif len(significant) > 1:
                res_regular_refine.append(significant)
            else:
                res_regular_refine.append(np.argmin(row))

        'Reject & Refine'
        res_reject_refine = []
        for i, row in enumerate(r_table_p):
            significant = np.where(row < alpha)[0]
            if len(significant) == 1:
                res_reject_refine.append(significant[0])
            elif len(significant) > 1:
                res_reject_refine.append(significant)
            else:
                res_reject_refine.append(-1)

        return res_regular, res_reject,res_regular_refine, res_reject_refine, r_table_p


    def checkR(self, X_train, y_train, X_test, y_test, target_list, method):
        """choose the r of rOP

        Args:
            train_X (ndarry): train_X
            train_y (array): train_y
            target_list (list): the class label list

        Returns:
            r (int): the best r according to the validation set
        """
        r_best = -1
        r_best_score = -1
        self.fit(X_train, y_train, method)
        info = np.ones((len(target_list), X_test.shape[0], X_test.shape[1]))
        for t in target_list:
            arr_t = np.ones((X_test.shape[0], X_test.shape[1]))
            for col in range(X_test.shape[1]):
                col_map = {}
                for indices, val in enumerate(X_test[:,col]):
                    if val not in col_map:
                        col_map[val] = []
                        col_map[val].append(indices)
                    else:
                        col_map[val].append(indices)
                for unique_value, appear_indices in col_map.items():
                    key = f'{t}-{col}-{unique_value}'
                    arr_t[appear_indices,col] = self.information.get(key,1)
            info[t] = np.sort(arr_t, axis=1)
        for r in range(X_test.shape[1]):
            r_table = info[:, :, r].T
            res = np.argmin(r_table, axis=1)
            if accuracy_score(y_test, res) > r_best_score:
                r_best = r
                r_best_score = accuracy_score(y_test, res)
        return r_best


    def store_features(self, dataset):
        """store features in a map so we can get one feature faster

        Args:
            dataset (dataframe): dataset

        Returns:
            map: store features, such as features[0] is the first column (feature)
        """
        features = {}
        num_columns = dataset.shape[1]
        for column_idx in range(num_columns):
            column_data = dataset.iloc[:, column_idx].astype(str)
            features[column_idx] = column_data
        return features

    def feature_subset_generation(self, train_x, valid_x, test_x, train_y, b1, b2):
        """generate feature subset

        Args:
            train_x (ndarry): train_X
            valid_x (ndarry): valid_X
            test_x (ndarry): test_X
            train_y (array): train_y
            num_features (int): the number of features
            b1 (int): number of iterations
            b2 (int): number of candidate subspaces in each round

        Returns:
            train_X, valid_X and test_X: the train_X, valid_X and test_X after feature subset generation
        """
        train_x = pd.DataFrame(train_x)
        valid_x = pd.DataFrame(valid_x)
        test_x = pd.DataFrame(test_x)
        num_features = test_x.shape[1]
        feature_map_train = self.store_features(train_x)
        feature_map_valid = self.store_features(valid_x)
        feature_map_test = self.store_features(test_x)
        new_subset_num = b1
        num_sample_train = len(train_x)
        num_sample_valid = len(valid_x)
        num_sample_test = len(test_x)
        best_name = []
        res_train_x = train_x.values
        res_valid_x = valid_x.values
        res_test_x = test_x.values
        for i in range(new_subset_num):
            new_array_train = np.empty((0, num_sample_train))
            new_array_valid = np.empty((0, num_sample_valid))
            new_array_test = np.empty((0, num_sample_test))
            random_list = []
            random_feature_name = []
            sample_indices = np.random.choice(num_sample_train, size=int(num_sample_train / 2), replace=False)
            for j in range(b2):
                subset_size = np.random.randint(2, min(num_features + 1,int(np.sqrt(num_sample_train)+1)))
                random_indices = sorted(np.random.choice(num_features, size=subset_size, replace=False))
                if random_indices in random_list:
                    continue
                columns_to_concat_train = [feature_map_train[k] for k in random_indices]
                columns_to_concat_valid = [feature_map_valid[k] for k in random_indices]
                columns_to_concat_test = [feature_map_test[k] for k in random_indices]
                concatenated_train = np.column_stack(columns_to_concat_train).sum(axis=1)
                concatenated_valid = np.column_stack(columns_to_concat_valid).sum(axis=1)
                concatenated_test = np.column_stack(columns_to_concat_test).sum(axis=1)
                new_array_train = np.vstack((new_array_train, concatenated_train))
                new_array_valid = np.vstack((new_array_valid, concatenated_valid))
                new_array_test = np.vstack((new_array_test, concatenated_test))
                random_list.append(random_indices)
                random_feature_name.append(random_indices)
            best_index = self.transform(new_array_train.T[sample_indices,:], train_y[sample_indices])
            if random_feature_name[best_index] not in best_name:
                best_name.append(random_feature_name[best_index])
                res_train_x = np.vstack((res_train_x.T, new_array_train.T[:, best_index])).T
                res_valid_x = np.vstack((res_valid_x.T, new_array_valid.T[:, best_index])).T
                res_test_x = np.vstack((res_test_x.T, new_array_test.T[:, best_index])).T
        return res_train_x, res_valid_x, res_test_x

    def transform(self, X, y):
        """apply relative risk method on the new b_2-feature-array to choose the best one subfeature

        Args:
            X (ndarry): the new b_2-feature-array of train_X
            y (array): the label of train_X
            class_list (array): the class list
            
        Returns:
            int: the index of the best feature
        """
        y_counter = Counter(y)
        selected_features = []
        for feature_idx in range(X.shape[1]):
            feature_idx_data = X[:, feature_idx]
            total_relative_risk = 0
            feature_map = self.process(feature_idx_data, y)
            for feature_value in feature_map.keys():
                total_show_up = feature_map[feature_value]['count']
                hit_labels = feature_map[feature_value]['labels']
                class_counts = Counter(hit_labels)
                majority_class_count = class_counts.most_common(1)[0][1]
                majority_class = class_counts.most_common(1)[0][0]

                contingency_table = np.zeros((2, 2))
                contingency_table[0, 0] = majority_class_count
                contingency_table[0, 1] = total_show_up - majority_class_count
                contingency_table[1, 0] = y_counter[majority_class] - majority_class_count
                contingency_table[1, 1] = len(y) - total_show_up - contingency_table[1, 0]
                contingency_table += 0.5
                relative_risk = (contingency_table[0, 0] / (contingency_table[0, 0] + contingency_table[0, 1])) / (
                        contingency_table[1, 0] / (contingency_table[1, 0] + contingency_table[1, 1]))

                total_relative_risk += relative_risk
            total_relative_risk /= len(feature_map.keys())
            selected_features.append(total_relative_risk)
        return np.argmax(selected_features)
    


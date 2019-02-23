from numpy import *
import matplotlib.pyplot as plt

import operator

def auto_norm(dataset):
    """将数据进行归一化,返回归一化之后的数据，每一列的数据范围， 最小值"""

    min_vals = dataset.min(0)
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals
    norm_data_set = zeros(shape(dataset))
    m = dataset.shape[0]
    norm_data_set = dataset - tile(min_vals, (m, 1))
    norm_data_set = norm_data_set/tile(ranges, (m, 1))
    return norm_data_set, ranges, min_vals

def file_2_matrix(filename):
    """将文件中的数据转换成数字数据"""

    fr = open(filename)
    array_lines = fr.readlines()
    number_of_lines = len(array_lines)
    return_mat = zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector

def classify0(in_x, dataset, labels, k):
    """knn算法"""

    dataset_size = dataset.shape[0]
    diff_mat = tile(in_x, (dataset_size, 1)) - dataset
    sq_diff_mat = diff_mat**2
    sq_distance = sq_diff_mat.sum(axis=1)
    distances = sq_distance**0.5 
    sorted_dist_indicies = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_dist_indicies[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                            key=operator.itemgetter(1), reverse=True)

    return sorted_class_count[0][0]

def dating_class_test():
    """测试数据，返回错误率"""

    ho_ratio = 0.1
    dating_data_mat, dating_labels = file_2_matrix('datingTestSet2.txt')
    normal_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = normal_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(normal_mat[i,:], normal_mat[num_test_vecs:m, :],
                    dating_labels[num_test_vecs:m], 3)
        print('the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i]))

        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print('the total errror rate is: %f' %  (error_count/float(num_test_vecs)))

def classify_person():
    """根据用户输入的信息进行预测"""

    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percent of time spent playing video games?"))
    ff_miles = float(input("Frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file_2_matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges, norm_mat, dating_labels, 3)
    
    print("You will probably like this person: ", result_list[classifier_result - 1])


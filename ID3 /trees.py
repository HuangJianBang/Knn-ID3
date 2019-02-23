from math import log
import operator

def calc_shannon_ent(dataset):
    """根据一个给定的数据集计算熵"""

    num_entyies = len(dataset)
    label_counts = {}
    for feat_vec in dataset:
        current_label = feat_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0       
    for key in label_counts:
        prob = float(label_counts[key]) / num_entyies
        shannon_ent -= prob * log(prob, 2)

    return shannon_ent

def split_dataset(dataset, axis, value):
    """根据属性，以及指定的属性的值划分数据集合"""

    ret_dataset = []
    for feat_vec in dataset:
        if feat_vec[axis] == value:
            reduce_feat_vec = feat_vec[:axis]
            reduce_feat_vec.extend(feat_vec[axis+1:])
            ret_dataset.append(reduce_feat_vec)

    return ret_dataset

def choose_best_feature_to_split(dataset):
    """选择最好的方式划分数据集，返回选中的属性"""

    num_features = len(dataset[0]) - 1  #2
    base_entropy = calc_shannon_ent(dataset)
    best_infogain = 0.0
    best_feature = -1

    for i in range(num_features):
        feat_list = [example[i] for example in dataset] #i=0[1, 1, 1, 0, 0]
        unique_vals = set(feat_list) # [1,0]
        new_entropy = 0.0

        for value in unique_vals:#value=1
            sub_dataset = split_dataset(dataset, i, value)#(dataset, 0,1) sub=[[1,yes],[1,'yes],[0,no]]
            prob = len(sub_dataset) / float(len(dataset))
            new_entropy += prob * calc_shannon_ent(sub_dataset)

        infogain = base_entropy - new_entropy
        
        if infogain > best_infogain:
            best_infogain = infogain
            best_feature = i

    return best_feature

def majority_cnt(class_list):
    """处理完所有属性之后，类标签不唯一，则选择值出现次数比较多的类"""

    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), 
                                key=operator.itemgetter(1),
                                reverse=True)
    return sorted_class_count[0][0] 

def create_tree(dataset, labels):
    """根据数据集和属性建立决策树"""

    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(dataset[0]) == 1:
        return majority_cnt(class_list)
    
    best_feature = choose_best_feature_to_split(dataset)
    best_feature_label = labels[best_feature]
    myTree = {best_feature_label:{}}
    del labels[best_feature]
    feat_values = [example[best_feature] for example in dataset]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        myTree[best_feature_label][value] = create_tree(split_dataset(
                                        dataset, best_feature, value),
                                        sub_labels
                                        )
    return myTree

def classify(input_tree, feat_labels, test_vec):
    """开始分类"""

    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    class_label = 'unknow'
    
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label

def store_tree(input_tree, filename):
    """将分类树存储起来，下次使用的时候就不用再次建立了"""
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input_tree, fw)
    fw.close()

def grab_tree(filename):
    """从文件加载已经建立好的树"""
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

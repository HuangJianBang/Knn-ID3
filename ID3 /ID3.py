import trees
import plot_tree

train_data_car = 'data/train_data_car.txt'
test_data_car = 'data/test_data_car.txt'
cache_car_tree = 'cache_tree/cache_car_tree'
train_labels_car = ['buying', 'maint',
                    'doors', 'persons', 'lug_boot', 'safety']
test_labels_car = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']


def create_store_tree():
    """根据训练集建立一棵树，并存入"""

    with open(train_data_car) as tdc:
        """打开训练数据文件，并读取数据"""
        train_cars = [inst.strip().split(',') for inst in tdc.readlines()]
        cars_tree = trees.create_tree(train_cars, train_labels_car)
        trees.store_tree(cars_tree, cache_car_tree)

def plot_a_tree():
    """绘制决策树"""
    plot_tree.create_plot(trees.grab_tree(cache_car_tree))

def testing_tree():
    """测试ID3分类算法的错误率"""

    with open(test_data_car) as tdc_2:
        test_cars = ([inst.strip().split(',') for inst in tdc_2.readlines()])

    error_count = 0
    for test_car in test_cars:
        label = trees.classify(trees.grab_tree(cache_car_tree),
                               test_labels_car, test_car[0:-1])
        if (label != test_car[-1]):
                error_count += 1

    print("%f" % (error_count/len(test_cars)))

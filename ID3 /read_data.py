import linecache
file_name = 'exam/data/car.txt'
test_data_car = 'exam/data/test_data_car.txt'

with open(test_data_car, 'w') as tdc:
    for line in range(3, 1729, 20):
        tdc.write(linecache.getline(file_name, line).strip() + '\n')





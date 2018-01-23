from numpy import array


# 1D example
data = [11, 22, 33, 44, 55]
# array of data
data_array = array(data)
print(data_array)
print(type(data_array))

# 2D example
data = [[11, 22, 33],
        [44, 55, 66],
        [77, 88, 99]]
# array
data_array = array(data)
print(data_array)
print(type(data_array))

print(data_array[1,1]) # 55
print(data_array[0,]) # [11 22 33]
print(data_array[2:, :1]) # [[77]]
print(data_array[0:3, 1]) # [22 55 88]


# split.. can be used to split test/train data
split = 2
train, test = data_array[:split,:], data[split:,:]
# using first two rows for training set, and 
# last row for the test set.


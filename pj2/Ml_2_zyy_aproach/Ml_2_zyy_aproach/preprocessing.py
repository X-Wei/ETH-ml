import numpy as np
from scipy import stats
from sklearn import preprocessing
from sklearn.decomposition import PCA
import tensorflow as tf
np.random.seed(1024)


dim = 7
n = 675
FourierDim = 3000
train_size = 575
label_num = 3
learning_rate = 0.001
seq = []
label = []
X = []
train_X = []
train_y = []
test_X = []
test_y = []
result_seq = []
validate_X = []


project_w =  np.random.standard_cauchy(size=(FourierDim, dim))
rotate_b = 2 * np.random.random_sample(FourierDim) * np.pi
rotate_b = rotate_b.reshape(FourierDim, 1)



def transform(x_original):
	newX = []
	for i in range(np.shape(x_original)[0]):
		newX.append(np.cos(np.dot(project_w, x_original[i].reshape(dim, 1)) + rotate_b) / np.sqrt(FourierDim/2))
	newX = np.squeeze(newX)
	return newX



def next_batch(arr, arr_y, num):
	idx = np.random.permutation(np.shape(arr)[0])
	arr_new = []
	arr_new_y = []
	counter = 0;
	for j in idx:
		if counter < num:
			arr_new.append(arr[j])
			arr_new_y.append(arr_y[j])
		else:
			break;
	return (arr_new, arr_new_y)

def transfer_label(label):
	label_M = np.eye(label_num).astype(int)
	return label_M[label]


# preprocessing training data
with open('train.csv','r') as trainfile:
	for line in trainfile:
		line = line.strip()
		(s, data_string) = line.split(",", 1)
		s = int(float(s))
		(x_string, l) = data_string.rsplit(",", 1)
		l = int(float(l))
		x_point = np.fromstring(x_string, sep=',')
		seq.append(s)
		label.append(l)
		X.append(x_point)
trainfile.close()

# preprocessing testing data
with open('validate_and_test.csv', 'r') as validatefile:
	for line in validatefile:
		line = line. strip()
		(s, x_string) = line.split(",", 1)
		s = int(float(s))
		x_point = np.fromstring(x_string, sep=',')
		result_seq.append(s)
		validate_X.append(x_point)
trainfile.close()


# fit pca
#pca = PCA(n_components=dim)
#pca.fit(np.concatenate((X, validate_X), axis = 0))

#X = np.squeeze(pca.transform(X))
X = stats.zscore(np.sqrt(X), axis=1)
#X = preprocessing.normalize(np.sqrt(X), norm='l1',axis=1)
X = transform(X)
idx = np.random.permutation(n)
counter = 0
for i in idx:
	if counter < train_size:
		train_X.append(X[i])
		train_y.append(transfer_label(label[i]))
	else:
		test_X.append(X[i])
		test_y.append(transfer_label(label[i]))
	counter += 1




#validate_X = np.squeeze(pca.transform(validate_X))
validate_X = stats.zscore(np.sqrt(validate_X), axis=1)
validate_X = transform(validate_X)


# change dimension
dim = FourierDim

# train model
x = tf.placeholder("float", [None, dim])
W = tf.Variable(tf.zeros([dim, label_num]))
b = tf.Variable(tf.zeros([label_num]))
y = tf.nn.softmax(tf.matmul(x, W)+b)  #define the model

y_ = tf.placeholder("float", [None, label_num])
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))  # define loss function

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()
# initialization
sess = tf.Session()
sess.run(init)
# train
for t in range(1000):
	(batch_xs, batch_ys) = next_batch(train_X, train_y, 100)
	sess.run(train_step, feed_dict={x: batch_xs, y_ : batch_ys})



# evaluation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print sess.run(accuracy, feed_dict={x: test_X, y_: test_y})

# predict
prediction = tf.argmax(y, 1)
result_y = sess.run(prediction, feed_dict={x: validate_X})

# output result 
with open('result.handin', 'w') as zyyfile:
	zyyfile.write('Id,Label\n')
	for i in range(np.shape(result_seq)[0]):
		zyyfile.write(str(result_seq[i]))
		zyyfile.write(',')
		zyyfile.write(str(result_y[i]))
		zyyfile.write('\n')
zyyfile.close()







import sys
sys.path.append("C:\ProjectDL")
from calcUtil import*
import glob
import random



##
# READING COUGH DATA
#
#
coughCloseList = glob.glob("./AudioData/Coughing/Close (cc)/*.wav")
coughDistantList = glob.glob("./AudioData/Coughing/Distant (cd)/*.wav")
coughAll = coughCloseList
coughAll.extend(coughDistantList)

listOfParticipantsToExclude = ["p05", "p17", "p34", "p20", "p28", "p09", "p08", "p11", "p31", "p21", "p14"]

testListCough = []
trainListCough = coughAll
for name in coughAll:
    for nameToExclude in listOfParticipantsToExclude:
        if nameToExclude in name:

            testListCough.append(name)
            trainListCough.remove(name)

print(np.size(testListCough))

##
# READING OTHER DATA
#
#

throat = glob.glob("./AudioData/Other Control Sounds/01_Throat Clearing/*.wav")
laughing =  glob.glob("./AudioData/Other Control Sounds/02_Laughing/*.wav")
speaking = glob.glob("./AudioData/Other Control Sounds/03_Speaking/*.wav")

other = throat
other.extend(laughing)
other.extend(speaking)

testListOther = []
trainListOther = other
for name in other:
    for nameToExclude in listOfParticipantsToExclude:
        if nameToExclude in name:
            testListOther.append(name)
            trainListOther.remove(name)

print(np.size(testListOther))

##
# BALANCING
#
#

SAMPLE_SIZE = 1000
TRAIN_SIZE = 0.8 * SAMPLE_SIZE
TEST_SIZE = 0.2 * SAMPLE_SIZE

random.seed(42)
train_random_coughs_numbers = random.sample(range(0,np.size(trainListCough)), int(TRAIN_SIZE/2))
train_random_other_numbers = random.sample(range(0,np.size(trainListOther)), int(TRAIN_SIZE/2))

test_random_coughs_numbers = random.sample(range(0,np.size(testListCough)), int(TEST_SIZE/2))
test_random_other_numbers = random.sample(range(0,np.size(testListOther)), int(TEST_SIZE/2))

trainFileNames = [trainListCough[x] for x in train_random_coughs_numbers]
trainFileNames.extend([trainListOther[x] for x in train_random_other_numbers])

testFileNames = [testListCough[x] for x in test_random_coughs_numbers]
testFileNames.extend([testListOther[x] for x in test_random_other_numbers])

#
# extracting features
#
#


tr_features,tr_labels = extract_features(trainFileNames)
tr_labels = one_hot_encode(tr_labels)


ts_features,ts_labels = extract_features(testFileNames)
ts_labels = one_hot_encode(ts_labels)

#
#Release the kraken
#
#
#

frames = 2
bands = 60

feature_size = 120 #frames x bands
num_labels = 2
num_channels = 2

batch_size = 50
kernel_size = 30
depth = 20
num_hidden = 200

learning_rate = 0.01
total_iterations = 2000


X = tf.placeholder(tf.float32, shape=[None,bands,frames,num_channels])
Y = tf.placeholder(tf.float32, shape=[None,num_labels])

cov = apply_convolution(X,kernel_size,num_channels,depth)

shape = cov.get_shape().as_list()
cov_flat = tf.reshape(cov, [-1, shape[1] * shape[2] * shape[3]])

f_weights = weight_variable([shape[1] * shape[2] * depth, num_hidden])
f_biases = bias_variable([num_hidden])
f = tf.nn.sigmoid(tf.add(tf.matmul(cov_flat, f_weights),f_biases))

out_weights = weight_variable([num_hidden, num_labels])
out_biases = bias_variable([num_labels])
y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases)

loss = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)

with tf.Session() as session:
    tf.initialize_all_variables().run()

    for itr in range(total_iterations):
        offset = (itr * batch_size) % (tr_labels.shape[0] - batch_size)
        batch_x = tr_features[offset:(offset + batch_size), :, :, :]
        batch_y = tr_labels[offset:(offset + batch_size), :]

        _, c = session.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
        cost_history = np.append(cost_history, c)

    print('Test accuracy: ', round(session.run(accuracy, feed_dict={X: ts_features, Y: ts_labels}), 3))
    fig = plt.figure(figsize=(15, 10))
    plt.plot(cost_history)
    plt.axis([0, total_iterations, 0, np.max(cost_history)])
    plt.show()
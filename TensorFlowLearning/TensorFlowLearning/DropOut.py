import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#载入数据集
mnist=input_data.read_data_sets('MNIST_data', one_hot=True)

#每个批次大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)

#创建一个简单的神经网络
#W = tf.Variable(tf.zeros([784,10]))
#b = tf.Variable(tf.zeros([10]))
#prediction = tf.nn.softmax(tf.matmul(x,W)+b)
W1=tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
L1_drop = tf.nn.dropout(L1,keep_prob)

W2=tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3=tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

W4=tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)


#二次代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵代价函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#结果存储在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax返回一维张量中最大值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#分配可用内存
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(31):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        test_acc = sess.run(accuracy , feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy , feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:1.0})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(test_acc)+",Training Accuracy"+str(train_acc))

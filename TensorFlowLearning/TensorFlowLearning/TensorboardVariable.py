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

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

#命名空间
with tf.name_scope('Input'):
    #定义两个placeholder
    x=tf.placeholder(tf.float32,[None,784],name = 'x-input')
    y = tf.placeholder(tf.float32, [None,10], name = 'y-input')

#创建一个简单的神经网络
with tf.name_scope('layer'):
    with tf.name_scope('wights'):
       W = tf.Variable(tf.zeros([784,10]))
       variable_summaries(W)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]))
        variable_summaries(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)


#二次代价函数
#loss = tf.reduce_mean(tf.square(y-prediction))
#交叉熵代价函数
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
    tf.summary.scalar('loss',loss)
#使用梯度下降法
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

#结果存储在一个布尔型列表中
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1)) #argmax返回一维张量中最大值所在的位置
    with tf.name_scope('accuracy'):
        #求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

#合并所有的summary
merged = tf.summary.merge_all()

#分配可用内存
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(31):        
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged, train_step], feed_dict={x:batch_xs,y:batch_ys})

        writer.add_summary(summary,epoch)
        test_acc = sess.run(accuracy , feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(test_acc))
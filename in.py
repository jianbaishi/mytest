#-*- coding:utf-8 -*-
'''
import tensorflow as tf
t1 = tf.constant([[1,2,3],[4,5,6]])
t2 = tf.constant([[7,8,9],[10,11,12]])
c1= tf.concat([t1,t2], 0)
c2= tf.concat([t1,t2], 1)
r1 = tf.reshape(c1, [-1])
r2 = tf.reshape(c2, [-1])

p1 = tf.stack([t1,t2],0)
p2 = tf.stack([t1,t2],1)

p3 = [t1, t2]

with tf.Session() as sess:
    print(sess.run([c1, c2]))
    print(sess.run([tf.rank(t1),tf.rank(c1)]))
    print("=======")
    print(sess.run([r1, r2]))
    print("=======X")
    print(sess.run([p1, p2]))
    print("=======X")
    print(sess.run([tf.rank(t1),tf.rank(p1)]))
    print(sess.run(tf.shape([p1, p2])))
    print("=======")
    print(sess.run(p3))
'''
'''
import tensorflow as tf
# 生成一个先入先出队列和一个QueueRunner,生成文件名队列
filenames = ['1.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# 定义Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# 定义Decoder
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value,record_defaults=record_defaults)
features = tf.stack([col1, col2, col3], 0)
label = tf.stack([col4,col5], 0)
example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=2, capacity=200, min_after_dequeue=100, num_threads=2)
# 运行Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。
    for i in range(10):
        e_val,l_val = sess.run([example_batch, label_batch])
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)

'''
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import os
'''
filename = os.path.join("data","contact")
filename_queue = tf.train.string_input_producer(["data/sh_hq_000001_2011.csv", 
                                                 "data/sh_hq_000001_2012.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)
produced = reader.num_records_produced()

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [['null'], ['null'], ['null'], ['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value,record_defaults=record_defaults)
features = tf.stack([col3, col1, col2, col4])
output = tf.stack([col5, col6, col7, col8, col9, col10, col11, col12, col13])
D_output = tf.to_double(output, name='ToDouble')

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)


  tmp_key = sess.run([key])
  print(tmp_key)
  while 1:
    # Retrieve a single instance:
    
    example, label = sess.run([features, D_output])
    date = col3.eval()
    print(produced.eval())
    print(col3)
    print(date)
    #print(example.eval(),label.eval())
    print(example,label)
    key_line = sess.run([key])
    #flag = tf.equal(tmp_key, key_line, name="Cmp")
    #if 1:
    #	break

  coord.request_stop()
  coord.join(threads)
'''
'''
import tensorflow as tf
import os

#设置工作目录

#查看目录
print(os.getcwd())

#读取函数定义
def read_data(file_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(file_queue)
    record_defaults = [['null'], ['null'], ['null'], ['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack([col3, col1, col2, col4])
    output = tf.stack([col5, col6, col7, col8, col9, col10, col11, col12, col13])
    D_output = tf.to_double(output, name='ToDouble')
    
    return features,D_output

def create_pipeline(batch_size, num_epochs=None):
    file_queue = tf.train.string_input_producer(["data/sh_hq_000001_2011.csv", 
                                                 "data/sh_hq_000001_2012.csv"], num_epochs=num_epochs)
    example, label = read_data(file_queue)

    min_after_dequeue = 100
    capacity = min_after_dequeue + batch_size
    print("capacity %d" %capacity)
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue
    )

    return example_batch, label_batch

x_train_batch, y_train_batch = create_pipeline(batch_size=10, num_epochs=100)
print(x_train_batch,y_train_batch)
'''
'''
with tf.Session() as sess:  
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
    for i in range(10):  
        e_val,l_val = sess.run([example_batch, label_batch])  
        print e_val,l_val  
    coord.request_stop()  
    coord.join(threads)  
    
'''
'''
import tensorflow as tf

# 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
filenames = ["data/sh_hq_000001_2011.csv", "data/sh_hq_000001_2012.csv"]  
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)  
# 定义Reader  
reader = tf.TextLineReader(skip_header_lines=1)  
key, value = reader.read(filename_queue)  
# 定义Decoder  
record_defaults = [['null'], ['null'], ['null'], ['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value,record_defaults=record_defaults)
features = tf.stack([col3, col1, col2, col4])
output = tf.stack([col5, col6, col7, col8, col9, col10, col11, col12, col13]) 
example_batch, label_batch = tf.train.shuffle_batch([features,output], batch_size=10, capacity=20, min_after_dequeue=10, num_threads=1)  
# 运行Graph  
with tf.Session() as sess:  
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
    for i in range(1):  
        e_val,l_val = sess.run([example_batch, label_batch])  
        print e_val,l_val  
    coord.request_stop()  
    coord.join(threads)  
'''
 
 
import tensorflow as tf
import numpy
import time

seq_max_len = 28
n_input = 28
rnn_unit = 28
n_hidden = 28 # hidden layer num of features
n_classes = 5 # MNIST total classes (0-9 digits)

seqlen = tf.placeholder(tf.int32, [None])

weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def get_file_line(filename_s):
    line_s = 0
    file_num = len(filename_s)
    for index in range(file_num):
        myfile = open(filename_s[index]) 
        line = len(myfile.readlines())
        line_s += line - 1
        print filename_s[index], line
    print line_s
    return line_s

def read(filenames, batch_size):
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
		
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    
    # 定义Reader  
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)  
    # 定义Decoder  
    record_defaults = [['null'], ['null'], ['null'], ['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack([col3, col1, col2, col4])
    features_1 = tf.stack([col5, col6, col7, col8, col9, col10, col11, col12, col13])
    in_put_f = tf.stack([col5, col6, col7, col8, col9, col10, col13])
    #features = tf.stack([[col3], [col1], [col2], [col4]])
    #features_1 = tf.stack([[col5], [col6], [col7], [col8], [col9], [col10], [col11], [col12], [col13]])
    #in_put_f = tf.stack([[col5], [col6], [col7], [col8], [col9], [col10], [col13]])
    out_put_f = tf.stack([col10])
    tf.set_random_seed(0)
    print(features, features_1, in_put_f, out_put_f)
    example_batch, label_batch, in_put, out_put = tf.train.batch([features, features_1, in_put_f, out_put_f], batch_size=batch_size, capacity=2000, num_threads=1)
    
    # 运行Graph  
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        e_val, l_val, in_val, out_val = sess.run([example_batch, label_batch, in_put, out_put])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return e_val, l_val, example_batch, label_batch, in_val, out_val, in_put, out_put

def get_info_form_file(filenames):
    lines = get_file_line(filename_s=filenames)
    e_val, l_val, example_batch, label_batch = read(filenames, lines)
    return e_val,l_val,example_batch,label_batch,in_val, out_val

#从文件读取信息
def get_info_form_file_and_slice(filenames, begin, size, diff_size):
    lines = get_file_line(filename_s=filenames)
    e_val_old, l_val_old, example_batch_old, label_batch_old, in_val_old, out_val_old, in_put_old, out_put_old = read(filenames, lines)
    
    example_batch = tf.slice(example_batch_old, [begin, 0], [size, 4])
    label_batch = tf.slice(label_batch_old, [begin, 0], [size, 9])
    in_put = tf.slice(in_put_old, [begin, 0], [size - diff_size, 7])
    out_put = tf.slice(out_put_old, [begin, 0], [size, 1])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        e_val,l_val, in_val, out_val = sess.run([example_batch, label_batch, in_put, out_put])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)

    return e_val, l_val, example_batch, label_batch, in_val, out_val, in_put, out_put

#输入数据合并
def get_in_concat_data(in1, in2, in3, in4):
    input_1 = tf.concat([in1, in2], 1)
    input_2 = tf.concat([input_1, in3], 1)
    input_3 = tf.concat([input_2, in4], 1)
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        input_3_val = sess.run(input_3)  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return input_3_val, input_3

def get_in_data_last5day(int_put, begin, size):
    int_put_data = tf.slice(int_put, [begin, 0], [size, 28])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        int_put_val = sess.run(int_put_data)  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return int_put_val, int_put_data

# 5day后的收盘价
def get_out_data_last5day(out, lines):

    #print(out_put_tra)
    end = lines - 5
    if end < 0:
        end = 0
        begin = 0
        index = 0
        tmp_index1 = begin + 1 + index
        tmp_index2 = begin + 2 + index
        tmp_index3 = begin + 3 + index
        tmp_index4 = begin + 4 + index
        tmp_index5 = begin + 5 + index
        print(lines, begin, index, tmp_index1)
        if tmp_index1 >= lines:
           tmp_index1 = lines - 1
        if tmp_index2 >= lines:
            tmp_index2 = tmp_index1
        if tmp_index3 >= lines:
            tmp_index3 = tmp_index2
        if tmp_index4 >= lines:
            tmp_index4 = tmp_index3
        if tmp_index5 >= lines:
            tmp_index5 = tmp_index4
        one_out = tf.gather(out,[tmp_index1, tmp_index2, tmp_index3, tmp_index4, tmp_index5])
        out_put_tra = tf.transpose(one_out)
    else:
        out_put = tf.gather(out,[1, 2, 3, 4, 5])
        out_put_tra = tf.transpose(out_put)

    print(lines, end)
    for index in range(1,end):
        one_out = tf.gather(out,[1 + index, 2 + index, 3 + index, 4 + index, 5 + index])
        one_out_tra = tf.transpose(one_out)
        out_put_tra = tf.concat([out_put_tra, one_out_tra], 0)
        #print(one_out)
        #print(one_out_tra)
    
    for index in range(end + 1, lines):
        begin = 0
        tmp_index1 = begin + 1 + index
        tmp_index2 = begin + 2 + index
        tmp_index3 = begin + 3 + index
        tmp_index4 = begin + 4 + index
        tmp_index5 = begin + 5 + index
        #print(lines, begin, index, tmp_index1)
        if tmp_index1 >= lines:
           tmp_index1 = lines - 1
        if tmp_index2 >= lines:
            tmp_index2 = tmp_index1
        if tmp_index3 >= lines:
            tmp_index3 = tmp_index2
        if tmp_index4 >= lines:
            tmp_index4 = tmp_index3
        if tmp_index5 >= lines:
            tmp_index5 = tmp_index4
        one_out = tf.gather(out,[tmp_index1, tmp_index2, tmp_index3, tmp_index4, tmp_index5])
        one_out_tra = tf.transpose(one_out)
        out_put_tra = tf.concat([out_put_tra, one_out_tra], 0)
        
    print(out_put_tra)
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        out_put_tra_val = sess.run(out_put_tra)
        print(out_put_tra_val)
        #print e_val,l_val  
        coord.request_stop()
        coord.join(threads)
    return out_put_tra_val, out_put_tra

def get_step_info(input, index, step, batch):
    return out;

def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    #print(x)
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    #tx2 = tf.transpose(tx3)
    #tx = tf.unstack(tx2, 1, 1)
    print(x)
    x = tf.unstack(x, 28, 1)
    #tx = tf.stack(x)
    #x_ = tf.transpose(tx)
    print(x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(28)

    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,n_input])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    print("input", input)
    #input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

def lstm_test(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    print(batch_size, time_step)
    w_in=weights['out']
    b_in=biases['out']  
    input=tf.reshape(X,[-1,n_input])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    print("input", input)
    #input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return X

def test_spahe(train_x):
    tx1 = tf.stack([train_x[1:5]])
    tx3 = tf.stack(train_x[1:2])
    tx2 = tf.transpose(tx3)
    tx = tf.unstack(tx2, 1, 1)
    #tx = tf.unstack(train_x[1:5], 28, 1)
    #tx2 = tf.depth_to_space(tx3, 2)
    with tf.Session() as sess:
        tx_ = sess.run(tx)
        tx1_ = sess.run(tx1)
        tx2_ = sess.run(tx2)
        print("||||||||||||||||||||||||||||||||||||||")
        print(tx1_)
        print("||||||||||||||||||||||||||||||||||||||")
        print(train_x[1:5])
        print(tx_)
        print(tx3, tx2)
        print(tx2_)
        print("||||||||||||||||||||||||||||||||||||||")
    return tx_

"""
    pred = dynamicRNN(X, seqlen, weights, biases)
    #print(train_x,train_y)
    #print(X,Y,pred)
    lr = 0.01
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    outfile="out/"
    #module_file = tf.train.latest_checkpoint(outfile)    
    module_file = tf.train.latest_checkpoint(outfile, latest_filename="train")
    a = tf.placeholder("float")
    b = tf.placeholder("float")
    c = tf.multiply(a, b)
    #T = lstm_test(X)
    #test,_ = lstm(train_x[0][0:0+2])
    #print("test", test)
"""

def train_lstm(train_x, train_y, batch_size=80,time_step=2,train_begin=0,train_end=50):
    X=tf.placeholder(tf.float32, shape=[None,28, 1])
    Y=tf.placeholder(tf.float32, shape=[None,5])
    batch_index = 10
    step = 2
    seqlen = 4
    #pred,state=lstm(X)
    #tmp_x = train_x[1:1+2]
    #tmp_y = train_y[1:1+2]
    #print(tmp_x, tmp_y)
    #tx = tf.unstack(tmp_x, 28, 1)
    #print("tx", tx)
    pred = dynamicRNN(X, seqlen, weights, biases)
    #print(train_x,train_y)
    #print(X,Y,pred)
    lr = 0.01
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    outfile="out/"
    #module_file = tf.train.latest_checkpoint(outfile)    
    module_file = tf.train.latest_checkpoint(outfile, latest_filename="train")
    a = tf.placeholder("float")
    b = tf.placeholder("float")
    c = tf.multiply(a, b)
    #T = lstm_test(X)
    #test,_ = lstm(train_x[0][0:0+2])
    #print("test", test)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #print(sess.run(c, feed_dict={a:2, b:5}))
        #tx_ = sess.run(tx)
        #print("---------")
        #print(tx)
        #print("---------")
        #print(tx_)
        #print("---------")
        #saver.restore(sess, module_file)
        #重复训练2000次
        for i in range(2):
            print(batch_size)
            batch_size = 5
            for step in range(batch_size):
                print(step)
                tmp_x = train_x[step:step+1]
                tmp_y = train_y[step:step+1]
                tx = tf.stack(tmp_x)
                x1 = tf.transpose(tx)
                x2 = tf.unstack(x1, 28, 0)
                x1_ = sess.run([x1])
                x2_ = sess.run(x2)
                #print(x1.ndims)
                #print("XXXXXXXXXXXXXXXX")
                #print(x1)
                #print(x2)
                #print(x1_)
                #print(x2_)
                #print("XXXXXXXXXXXXXXXX")
                #print(tmp_x, tmp_y)
                #print("XXXXXXXXXXXXXXXX")
                #pred_ = sess.run([pred],feed_dict={X:x1_})
                #print(pred_)
                #print(step, "=", train_x[0][step:step+2])
                #t_y = sess.run([tf.slice(train_y, [step, 0], [step + 1, 5])])
                #loss_ = sess.run([loss],feed_dict={X:tmp_x,Y:tmp_y})
                #train_op_ = sess.run([train_op],feed_dict={X:train_x[0][step:step+1],Y:train_y[0][step:step+1]})
                _,loss_=sess.run([train_op,loss],feed_dict={X:x1_,Y:tmp_y})
                #print(loss_)
                #print(step, t_x, t_y)
                #_,loss_=sess.run([train_op,loss],feed_dict={X:train_x,Y:train_y})
            print(i,loss_)
            print("保存模型：",saver.save(sess,'stock2.model',global_step=i))
    return train_x

def pred_test(test_x, test_y, test_size, time_step=20):
    x=tf.placeholder(tf.float32, shape=[None,28, 1])
    y=tf.placeholder(tf.float32, shape=[None,5])
    seqlen = 4
    pred = dynamicRNN(x, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    
    saver=tf.train.Saver(tf.global_variables())
    outfile="out/"
    #module_file = tf.train.latest_checkpoint(outfile)    
    module_file = tf.train.latest_checkpoint(outfile, latest_filename="train")
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, module_file) 
        for step in range(test_size):
            tmp_x = test_x[step:step+1]
            tmp_y = test_y[step:step+1]
            tx = tf.stack(tmp_x)
            x1 = tf.transpose(tx)
            x1_ = sess.run([x1])
            acc = sess.run(accuracy, feed_dict={x: x1_, y: tmp_y})
            print(acc)
    return acc


def prediction(test_x, test_size, time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)     
    saver=tf.train.Saver(tf.global_variables())
    module_file = os.path.join([tf.train.latest_checkpoint(outfile), "meta"])
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint()
        saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(test_size):
          tmp_x = test_x[step:step+1]
          tx = tf.stack(tmp_x)
          x1 = tf.transpose(tx)
          x1_ = sess.run([x1])
          prob=sess.run(pred,feed_dict={X:x1_})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)]) #acc为测试集偏差
    return acc

#numpy.set_printoptions(threshold='nan')

filenames_000001 = ["data/sh_hq_000001_2011.csv", "data/sh_hq_000001_2012.csv", "data/sh_hq_000001_2013.csv", "data/sh_hq_000001_2014.csv", \
    "data/sh_hq_000001_2015.csv", "data/sh_hq_000001_2016.csv", "data/sh_hq_000001_2017.csv"]
filenames_399001 = ["data/sz_hq_399001_2011.csv", "data/sz_hq_399001_2012.csv", "data/sz_hq_399001_2013.csv", "data/sz_hq_399001_2014.csv", \
    "data/sz_hq_399001_2015.csv", "data/sz_hq_399001_2016.csv", "data/sz_hq_399001_2017.csv"]
filenames_399006 = ["data/sz_hq_399006_2011.csv", "data/sz_hq_399006_2012.csv", "data/sz_hq_399006_2013.csv", "data/sz_hq_399006_2014.csv", \
    "data/sz_hq_399006_2015.csv", "data/sz_hq_399006_2016.csv", "data/sz_hq_399006_2017.csv"]
filenames_300188 = ["data/sz_hq_300188_2011.csv", "data/sz_hq_300188_2012.csv", "data/sz_hq_300188_2013.csv", "data/sz_hq_300188_2014.csv", \
    "data/sz_hq_300188_2015.csv", "data/sz_hq_300188_2016.csv", "data/sz_hq_300188_2017.csv"]

lines_000001 = get_file_line(filename_s=filenames_000001)
lines_300188 = get_file_line(filename_s=filenames_300188)
begin = lines_000001 - lines_300188 + 1
#后面5个数据无法测试
test_day = 5
input_size = lines_300188 - 1
output_size = input_size

#e_val_000001, l_val_000001, example_batch_000001, label_batch_000001 = get_info_form_file(filenames_000001)
#e_val_399001, l_val_399001, example_batch_399001, label_batch_399001 = get_info_form_file(filenames_399001)
#e_val_399006, l_val_399006, example_batch_399006, label_batch_399006 = get_info_form_file(filenames_399006)
#e_val_300188, l_val_300188, example_batch_300188, label_batch_300188 = get_info_form_file(filenames_300188)

#300188 过滤第一个
e_val_000001, l_val_000001, example_batch_000001, label_batch_000001, in_val_000001, \
    out_val_000001, in_put_000001, out_put_000001 = \
    get_info_form_file_and_slice(filenames=filenames_000001, begin=begin, size=input_size, diff_size=0)
e_val_399001, l_val_399001, example_batch_399001, label_batch_399001, in_val_399001, \
    out_val_399001, in_put_399001, out_put_399001 = \
    get_info_form_file_and_slice(filenames=filenames_399001, begin=begin, size=input_size, diff_size=0)
e_val_399006, l_val_399006, example_batch_399006, label_batch_399006, in_val_399006, \
    out_val_399006, in_put_399006, out_put_399006 = \
    get_info_form_file_and_slice(filenames=filenames_399006, begin=begin, size=input_size, diff_size=0)
e_val_300188, l_val_300188, example_batch_300188, label_batch_300188, in_val_300188, \
    out_val_300188, in_put_300188, out_put_300188 = \
    get_info_form_file_and_slice(filenames=filenames_300188, begin=1, size=input_size, diff_size=0)

#print(e_val_000001, l_val_000001)
#print(e_val_399001, l_val_399001)
#print(e_val_399006, l_val_399006)
#print(e_val_300188, l_val_300188)
#print(in_val_300188, out_val_300188)
print(in_put_000001, out_put_000001)
print(in_put_399001, out_put_399001)
print(in_put_399006, out_put_399006)
print(in_put_300188, out_put_300188)

input_val, input_data = get_in_concat_data(in1=in_put_000001, in2=in_put_399001, in3=in_put_399006, in4=in_put_300188)
print(input_data)
print("XXXXXXXXXXXXXXXX")
print(input_val)
print(input_val.shape)
print("XXXXXXXXXXXXXXXX")

real_input_val, real_input_data = get_in_data_last5day(input_data, 0, input_size)
print(real_input_data)

print("=======================")
print(out_put_300188)
output_data_val, output_data = get_out_data_last5day(out_put_300188, output_size)
print(output_data_val.shape)
print(output_data_val[2:4])
print("=======================")
#print(out_val_300188)

#测试数据获取

filenames_000001_test = ["data/sh_hq_000001_2017_test.csv"]
filenames_399001_test = ["data/sz_hq_399001_2017_test.csv"]
filenames_399006_test = ["data/sz_hq_399006_2017_test.csv"]
filenames_300188_test = ["data/sz_hq_300188_2017_test.csv"]

lines_000001_test = get_file_line(filename_s=filenames_000001_test)
lines_300188_test = get_file_line(filename_s=filenames_300188_test)
begin_test = 0
#后面5个数据无法测试
test_day_test = 5
input_size_test = lines_300188_test
output_size_test = input_size_test

#e_val_000001, l_val_000001, example_batch_000001, label_batch_000001 = get_info_form_file(filenames_000001)
#e_val_399001, l_val_399001, example_batch_399001, label_batch_399001 = get_info_form_file(filenames_399001)
#e_val_399006, l_val_399006, example_batch_399006, label_batch_399006 = get_info_form_file(filenames_399006)
#e_val_300188, l_val_300188, example_batch_300188, label_batch_300188 = get_info_form_file(filenames_300188)

e_val_000001_test, l_val_000001_test, example_batch_000001_test, label_batch_000001_test, in_val_000001_test, \
    out_val_000001_test, in_put_000001_test, out_put_000001_test = \
    get_info_form_file_and_slice(filenames=filenames_000001_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_399001_test, l_val_399001_test, example_batch_399001_test, label_batch_399001_test, in_val_399001_test, \
    out_val_399001_test, in_put_399001_test, out_put_399001_test = \
    get_info_form_file_and_slice(filenames=filenames_399001_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_399006_test, l_val_399006_test, example_batch_399006_test, label_batch_399006_test, in_val_399006_test, \
    out_val_399006_test, in_put_399006_test, out_put_399006_test = \
    get_info_form_file_and_slice(filenames=filenames_399006_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_300188_test, l_val_300188_test, example_batch_300188_test, label_batch_300188_test, in_val_300188_test, \
    out_val_300188_test, in_put_300188_test, out_put_300188_test = \
    get_info_form_file_and_slice(filenames=filenames_300188_test, begin=begin_test, size=input_size_test, diff_size=0)


print(in_put_000001_test, out_put_000001_test)
print(in_put_399001_test, out_put_399001_test)
print(in_put_399006_test, out_put_399006_test)
print(in_put_300188_test, out_put_300188_test)

input_val_test, input_data_test = get_in_concat_data(in1=in_put_000001_test, in2=in_put_399001_test, in3=in_put_399006_test, in4=in_put_300188_test)
print(input_data_test)
print(input_val_test)

real_input_val_test, real_input_data_test = get_in_data_last5day(input_data_test, 0, input_size_test)
print(real_input_data_test)

print("=======================")
output_data_val_test, output_data_test = get_out_data_last5day(out_put_300188_test, output_size_test)
print(output_data_test)
print("=======================")



#X=tf.placeholder(tf.float32, shape=[None,10,28])
#pred,states=lstm(X)
#print(X, pred, states)
#_ = train_lstm(train_x = real_input_val, train_y = output_data_val, batch_size = input_size - 5)
#_ = train_lstm(train_x = real_input_val, train_y = output_data_val, batch_size = input_size - 5)
#_ = test_spahe(train_x = real_input_val)
#_ = pred_test(test_x = real_input_val, test_y = output_data_val, test_size = 50, time_step=20)
#_ = pred_test(test_x = real_input_val_test, test_y = output_data_val_test, test_size = 50, time_step=20)

input_state=sys.argv[1]

train_x = real_input_val
train_y = output_data_val
test_x = real_input_val_test
test_y = output_data_val_test

#训练
x=tf.placeholder(tf.float32, shape=[None,28, 1])
y=tf.placeholder(tf.float32, shape=[None,5])
batch_index = 10
step = 2
seqlen = 4

pred = dynamicRNN(x, seqlen, weights, biases)
lr = 0.1
#损失函数
loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1])))
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)

# Evaluate model
out_x = tf.argmax(pred,1)
out_y = tf.argmax(y,1)
correct_pred = tf.equal(out_x, out_y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
checkpoint_dir="train/"
#module_file = tf.train.latest_checkpoint(outfile, latest_filename="train")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("restore:")
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    if input_state == '0':
#        print("restore:")
#        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#        if ckpt and ckpt.model_checkpoint_path:
#            saver.restore(sess, ckpt.model_checkpoint_path)
#    else:
        print("Train:")
        #重复训练2000次
        for i in range(2):
            #print(batch_size)
            batch_size = input_size - 5
            print(input_size, batch_size)
            for step in range(batch_size):
                tmp_x = train_x[step:step+1]
                tmp_y = train_y[step:step+1]
                tx = tf.stack(tmp_x)
                x1 = tf.transpose(tx)
                x1_ = sess.run([x1])
                #print(x1_, tmp_y)
                #loss_ = sess.run(optimizer, feed_dict={x:x1_,y:tmp_y})
                _,loss_=sess.run([train_op,loss],feed_dict={x:x1_,y:tmp_y})
                print(step, loss_)
                if step % 100 == 0:
                    pred_ = sess.run(pred, feed_dict={x: x1_})
                    print("#input: ", tmp_x)
                    print("#output: ", tmp_y)
                    print("#pred: ", pred_)
            print(i,loss_)
            filename=checkpoint_dir + 'train.model'
            print("Save file：",saver.save(sess,filename,global_step=int(time.time())))

#测试

    test_size = input_size_test
    for step in range(test_size):
        tmp_test_x = test_x[step:step+1]
        tmp_test_y = test_y[step:step+1]
        test_tx = tf.stack(tmp_test_x)
        test_x1 = tf.transpose(test_tx)
        test_x1_ = sess.run([test_x1])
        pred_ = sess.run(pred, feed_dict={x: test_x1_})
        out_y_ = sess.run(out_y, feed_dict={y: tmp_test_y})
        print("-------------------------")
        print(tmp_test_x)
        print(tmp_test_y)
        print(pred_)
        print("-------------------------")
        acc = sess.run(accuracy, feed_dict={x: test_x1_, y: tmp_test_y})
        #loss_ = sess.run(optimizer, feed_dict={x: test_x1_, y: tmp_test_y})
        _,loss_=sess.run([train_op,loss],feed_dict={x: test_x1_, y: tmp_test_y})
        #print(acc)

#    for step in range(input_size_test):
#        tmp_test_x = test_x[step:step+1]
#        test_tx = tf.stack(tmp_test_x)
#        test_x1 = tf.transpose(test_tx)
#        test_x1_ = sess.run([test_x1])
#        pred_ = sess.run(pred, feed_dict={x: test_x1_})
#        print("**************************")
#        print(tmp_test_x)
#        print(pred_)
#        print("**************************")
        #acc = sess.run(accuracy, feed_dict={x: test_x1_, y: tmp_test_y})
        #print(acc)

'''
in_1 = tf.concat([in_put_000001, in_put_399001], 1)
in_2 = tf.concat([in_1, in_put_399006], 1)
in_3 = tf.concat([in_2, in_put_300188], 1)
#in_1 = tf.tile(in_put_000001, in_put_399001, name=None)
print(in_1)
print(in_2)
print(in_3)

with tf.Session() as sess: 
    coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
    threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
    in3_val = sess.run([in_3]) 
    print(in3_val)
    coord.request_stop()  
    coord.join(threads)
'''

'''
lines_000001 = get_file_line(filename_s=filenames_000001)


lines_000001 = get_file_line(filename_s=filenames_000001)
val1_000001, val2_000001 = read(filenames_000001, lines_000001)
print("================= 000001 =================")
print(val1_000001, val2_000001)
print("=================  end   =================")


lines_399001 = get_file_line(filename_s=filenames_399001)
val1_399001, val2_399001 = read(filenames_399001, lines_399001)
print("================= 399001 =================")
#print(val1_399001, val2_399001)
print("=================  end   =================")


lines_399006 = get_file_line(filename_s=filenames_399006)
val1_399006, val2_399006 = read(filenames_399006, lines_399006)
print("================= 399006 =================")
#print(val1_399006, val2_399006)
print("=================  end   =================")


lines_300188 = get_file_line(filename_s=filenames_300188)
val1_300188, val2_300188 = read(filenames_300188, lines_300188)
print("================= 300188 =================")
#print(val1_300188, val2_300188)
print(val1_300188[100], val2_300188[100])
print("=================  end   =================")
'''


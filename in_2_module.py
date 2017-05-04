#-*- coding:utf-8 -*-
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import os
import numpy
import time

seq_max_len = 24
n_input = 24
rnn_unit = 24
n_hidden = 24 # hidden layer num of features
n_classes = 1 # MNIST total classes (0-9 digits)

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
    print(filename_s, line_s)
    return line_s

def read(filenames, batch_size):
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列

    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

    # 定义Reader  
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)  
    # 定义Decoder  
    record_defaults = [['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7 = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack([col1])
    features_1 = tf.stack([col2, col3, col4, col5, col6, col7])
    in_put_f = tf.stack([col2, col3, col4, col5, col6, col7])
    #features = tf.stack([[col3], [col1], [col2], [col4]])
    #features_1 = tf.stack([[col5], [col6], [col7], [col8], [col9], [col10], [col11], [col12], [col13]])
    #in_put_f = tf.stack([[col5], [col6], [col7], [col8], [col9], [col10], [col13]])
    out_put_f = tf.stack([col5])
    tf.set_random_seed(0)

    #example_batch, label_batch, in_put, out_put = tf.train.shuffle_batch([features, features_1, in_put_f, out_put_f], batch_size=batch_size, capacity=8000, min_after_dequeue=1000)
    example_batch, label_batch, in_put, out_put = tf.train.batch([features, features_1, in_put_f, out_put_f], batch_size=batch_size, capacity=20000, num_threads=1)
    #example_batch, label_batch, in_put, out_put = tf.train.batch_join([features, features_1, in_put_f, out_put_f], batch_size=batch_size, capacity=2000)
    
    # 运行Graph  
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        #e_val, l_val, in_val, out_val = sess.run([example_batch, label_batch, in_put, out_put])
        e_val, l_val, in_val, out_val = sess.run([example_batch, label_batch, in_put_f, out_put_f])
        #out_val = sess.run([out_put])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
     
    #print(e_val)
    return e_val, example_batch, in_val, out_val, in_put, out_put

def get_info_form_file(filenames):
    lines = get_file_line(filename_s=filenames)
    e_val, l_val, example_batch, label_batch = read(filenames, lines)
    return e_val,l_val,example_batch,label_batch,in_val, out_val

#从文件读取信息
def get_info_form_file_and_slice(filenames, begin, size, diff_size):
    lines = get_file_line(filename_s=filenames)
    e_val_old, example_batch_old, in_val_old, out_val_old, in_put_old, out_put_old = read(filenames, lines)
    #print(example_batch_old, label_batch_old,	 in_put_old, out_put_old)
    #size = lines - 1
    #print(e_val_old)
    example_batch = tf.slice(example_batch_old, [begin, 0], [size, 1])
    in_put = tf.slice(in_put_old, [begin, 0], [size - diff_size, 6])
    out_put = tf.slice(out_put_old, [begin, 0], [size, 1])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        #print(begin, size, diff_size)
        #e_val = sess.run([example_batch])
        e_val,in_val, out_val = sess.run([example_batch, in_put, out_put])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)

    print(e_val)
    return e_val, example_batch, in_val, out_val, in_put, out_put


def fix_input_data(input, len, input_comp, comp, comp_len): 
    for comp_index in range(comp_len):
        #print(comp[comp_index][0])
        if input_comp[0][0] == comp[comp_index][0]:
            break
    tmp_comp_len = comp_index
    index = 1
    output = input[0]
    count = 1
    for comp_index in range(tmp_comp_len + 1, comp_len):
        #print(input_comp[index], comp[comp_index])
        if input_comp[index][0] == comp[comp_index][0]:
            #print(comp[comp_index][0], 	input[index])
            output  = tf.concat([output, 	input[index]], 0)
            index = index + 1
        else:
            tmp_input = [input[index][3], input[index][3], input[index][3], input[index][3], 0, 0]
            #print(comp[comp_index][0], 	tmp_input)
            output  = tf.concat([output, tmp_input], 0)
        count = count + 1
    
    tmp_output = tf.reshape(output, [-1, 6])
    with tf.Session() as sess:
        output_val = sess.run(tmp_output)
    print(output_val, tmp_output)
    return output_val, tmp_output, count


#输入数据合并
def get_in_concat_data(in1, in2, in3, in4, len):
    tmp_len = tf.size(in1) / 6
    tmp_in1 = tf.slice(in1, [tmp_len - len, 0], [len, 6])

    tmp_len = tf.size(in2) / 6
    tmp_in2 = tf.slice(in2, [tmp_len - len, 0], [len, 6])

    tmp_len = tf.size(in3) / 6
    tmp_in3 = tf.slice(in3, [tmp_len - len, 0], [len, 6])
    
    input_1 = tf.concat([tmp_in1, tmp_in2], 1)
    input_2 = tf.concat([input_1, tmp_in3], 1)
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
    int_put_data = tf.slice(int_put, [begin, 0], [size, 24])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        int_put_val = sess.run(int_put_data)  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return int_put_val, int_put_data

def get_out_data_last5day_new(input, size):
    print(size)
    for index in range(size):
        t_index1 = index + 1
        if t_index1 >= size:
            t_index1 = t_index1 - 1

        t_index2 = index + 2
        if t_index2 >= size:
            t_index2 = t_index1

        t_index3 = index + 3
        if t_index3 >= size:
            t_index3 = t_index2

        t_index4 = index + 4
        if t_index4 >= size:
            t_index4 = t_index3

        t_index5 = index + 5
        if t_index5 >= size:
            t_index5 = t_index4
        
        #print(t_index1,t_index2,t_index3,t_index4,t_index5)
        if index == 0:
            output = [input[t_index1][3], input[t_index2][3], input[t_index3][3], input[t_index4][3], input[t_index5][3]]
        else:
            tmp_input = [input[t_index1][3], input[t_index2][3], input[t_index3][3], input[t_index4][3], input[t_index5][3]]
            output  = tf.concat([output, tmp_input], 0)
    #print(output)
    tmp_output = tf.reshape(output, [-1, 5])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        output_val = sess.run(tmp_output)
        coord.request_stop()  
        coord.join(threads)

    #print(output_val, tmp_output)
    return output_val, tmp_output

def get_out_data_1(output_data, fix_len):
    tmp_output = tf.slice(output_data, [0, 0], [fix_len, 1])
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        output_val = sess.run(tmp_output)
        coord.request_stop()  
        coord.join(threads)

    #print(output_val, tmp_output)
    return output_val, tmp_output

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

def get_test_data(x, y, begin, size):
    in_x = tf.slice(x, [begin, 0], [size, 24])
    in_y = tf.slice(y, [begin, 0], [size, n_classes])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        x_val, y_val = sess.run([in_x, in_y])
        #print e_val,l_val  
        coord.request_stop()
        coord.join(threads)
    return x_val, y_val

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
    #print(x)
    x = tf.unstack(x, rnn_unit, 1)
    #tx = tf.stack(x)
    #x_ = tf.transpose(tx)
    #print(x)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(rnn_unit)

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
    X=tf.placeholder(tf.float32, shape=[None,rnn_unit, 1])
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
                x2 = tf.unstack(x1, rnn_unit, 0)
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
    x=tf.placeholder(tf.float32, shape=[None,rnn_unit, 1])
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

filenames_000001 = ["data1/000001.csv"]
filenames_399001 = ["data1/399001.csv"]
filenames_399006 = ["data1/399006.csv"]
filenames_300188 = ["data1/300188.csv"]

lines_000001 = get_file_line(filename_s=filenames_000001)
lines_399001 = get_file_line(filename_s=filenames_399001)
lines_399006 = get_file_line(filename_s=filenames_399006)
lines_300188 = get_file_line(filename_s=filenames_300188)
#begin = lines_000001 - lines_300188 + 1
#后面5个数据无法测试
test_day = 5
input_size = lines_300188
output_size = input_size

#e_val_000001, l_val_000001, example_batch_000001, label_batch_000001 = get_info_form_file(filenames_000001)
#e_val_399001, l_val_399001, example_batch_399001, label_batch_399001 = get_info_form_file(filenames_399001)
#e_val_399006, l_val_399006, example_batch_399006, label_batch_399006 = get_info_form_file(filenames_399006)
#e_val_300188, l_val_300188, example_batch_300188, label_batch_300188 = get_info_form_file(filenames_300188)

#300188 过滤第一个
begin = 0 #lines_000001 - lines_300188
e_val_000001, example_batch_000001, in_val_000001, \
    out_val_000001, in_put_000001, out_put_000001 = \
    get_info_form_file_and_slice(filenames=filenames_000001, begin=begin, size=lines_000001, diff_size=0)

#begin = lines_399001 - lines_300188
e_val_399001, example_batch_399001, in_val_399001, \
    out_val_399001, in_put_399001, out_put_399001 = \
    get_info_form_file_and_slice(filenames=filenames_399001, begin=begin, size=lines_399001, diff_size=0)

#begin = lines_399006 - lines_300188
e_val_399006, example_batch_399006, in_val_399006, \
    out_val_399006, in_put_399006, out_put_399006 = \
    get_info_form_file_and_slice(filenames=filenames_399006, begin=begin, size=lines_399006, diff_size=0)

#begin = lines_399001 - lines_300188
e_val_300188, example_batch_300188, in_val_300188, \
    out_val_300188, in_put_300188, out_put_300188 = \
    get_info_form_file_and_slice(filenames=filenames_300188, begin=0, size=input_size, diff_size=0)

fix_300188_val, fix_300188, fix_len = fix_input_data(in_val_300188, lines_300188 - 1, e_val_300188, e_val_000001, lines_000001)

#print(e_val_000001, l_val_000001)
#print(e_val_399001, l_val_399001)
#print(e_val_399006, l_val_399006)
#print(e_val_300188, l_val_300188)
#print(in_val_300188, out_val_300188)
print(in_put_000001, out_put_000001)
print(in_put_399001, out_put_399001)
print(in_put_399006, out_put_399006)
print(in_put_300188, out_put_300188)

print(fix_300188_val, fix_300188)

input_val, input_data = get_in_concat_data(in1=in_put_000001, in2=in_put_399001, in3=in_put_399006, in4=fix_300188, len = fix_len)
print(input_data)
print("XXXXXXXXXXXXXXXX")
print(input_val)
print(input_val.shape)
print("XXXXXXXXXXXXXXXX")

real_input_val, real_input_data = get_in_data_last5day(input_data, 0, fix_len)
print(real_input_data)

print("=======================")
print(out_put_300188)
output_data_val, output_data = get_out_data_last5day_new(fix_300188_val, fix_len)
real_output_data_val, real_output_data = get_out_data_1(output_data, fix_len)
#output_data_val, output_data = get_out_data_last5day(fix_300188, fix_len)
print(output_data_val.shape)
print(output_data, real_output_data)
print(output_data_val)
print(real_output_data_val)
print("=======================")
print("=======================")
#print(out_val_300188)
print(real_input_val)
print(output_data_val)
print("=======================")
print("=======================")

#测试数据获取

filenames_000001_test = ["data1/000001_test.csv"]
filenames_399001_test = ["data1/399001_test.csv"]
filenames_399006_test = ["data1/399006_test.csv"]
filenames_300188_test = ["data1/300188_test.csv"]

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

e_val_000001_test, example_batch_000001_test, in_val_000001_test, \
    out_val_000001_test, in_put_000001_test, out_put_000001_test = \
    get_info_form_file_and_slice(filenames=filenames_000001_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_399001_test, example_batch_399001_test, in_val_399001_test, \
    out_val_399001_test, in_put_399001_test, out_put_399001_test = \
    get_info_form_file_and_slice(filenames=filenames_399001_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_399006_test, example_batch_399006_test, in_val_399006_test, \
    out_val_399006_test, in_put_399006_test, out_put_399006_test = \
    get_info_form_file_and_slice(filenames=filenames_399006_test, begin=begin_test, size=input_size_test, diff_size=0)
e_val_300188_test, example_batch_300188_test, in_val_300188_test, \
    out_val_300188_test, in_put_300188_test, out_put_300188_test = \
    get_info_form_file_and_slice(filenames=filenames_300188_test, begin=begin_test, size=input_size_test, diff_size=0)


print(in_put_000001_test, out_put_000001_test)
print(in_put_399001_test, out_put_399001_test)
print(in_put_399006_test, out_put_399006_test)
print(in_put_300188_test, out_put_300188_test)
print(input_size_test)
input_val_test, input_data_test = get_in_concat_data(in1=in_put_000001_test, in2=in_put_399001_test, in3=in_put_399006_test, in4=in_put_300188_test, len=input_size_test)
print(input_data_test)
print(input_val_test)

real_input_val_test, real_input_data_test = get_in_data_last5day(input_data_test, 0, input_size_test)
print(real_input_data_test)

print("=======================")
output_data_val_test, output_data_test = get_out_data_last5day_new(in_put_300188_test, output_size_test)
real_output_data_val_test, real_output_data_test = get_out_data_1(in_put_300188_test, output_size_test)

#output_data_val_test, output_data_test = get_out_data_last5day(out_put_300188_test, output_size_test)
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
""" 
# 5个预测数据
train_x = real_input_val
train_y = output_data_val
test_x = real_input_val_test
test_y = output_data_val_test
"""
#一个预测数据
train_x = real_input_val
train_y = real_output_data_val
test_x = real_input_val_test
test_y = real_output_data_val_test

#训练
x=tf.placeholder(tf.float32, shape=[None,rnn_unit, 1])
y=tf.placeholder(tf.float32, shape=[None,n_classes])
batch_index = 10
step = 2
seqlen = 4

pred = dynamicRNN(x, seqlen, weights, biases)
t_pred = tf.slice(pred, [0, 0], [1, 1])
t_y = tf.slice(y, [0, 0], [1, 1])
lr = 0.01
#损失函数
#loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(y, [-1])))
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
checkpoint_dir="train1/"
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
        for i in range(50):
            #print(batch_size)
            batch_size = fix_len - 5
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
                if step % 100 == 0 or step >= batch_size - 1:
                    pred_ = sess.run(pred, feed_dict={x: x1_})
                    print("#input: ", tmp_x)
                    print("#output: ", tmp_y)
                    print("#pred: ", pred_)
                    t_val_x, t_val_y = sess.run([t_pred, t_y],feed_dict={x:x1_,y:tmp_y})
                    print(t_val_x, t_val_y)
            print(i,loss_)
            filename=checkpoint_dir + 'train.model'
            print("Save file：",saver.save(sess,filename,global_step=int(time.time())))

#测试
    test_size = 5
    test_x, test_y = get_test_data(train_x, train_y, fix_len - test_size, test_size)
    print(test_x, test_y)

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


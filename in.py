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
import tensorflow as tf
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
    out_put_f = tf.stack([col10])
    tf.set_random_seed(0)
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
        input_3_val = sess.run([input_3])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return input_3_val, input_3

def get_in_data_last5dat(int_put, begin, size):
    int_put_data = tf.slice(int_put, [begin, 0], [size, 4])
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        int_put_val = sess.run([int_put_data])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return int_put_val, int_put_data

# 5day后的收盘价
def get_out_data_last5day(out, lines):
    out_put = tf.gather(out,[1, 2, 3, 4, 5])
    out_put_tra = tf.transpose(out_put)
    print(out_put_tra)
    for index in range(1,lines - 5):
        one_out = tf.gather(out,[1 + index, 2 + index, 3 + index, 4 + index, 5 + index])
        one_out_tra = tf.transpose(one_out)
        out_put_tra = tf.concat([out_put_tra, one_out_tra], 0)
        #print(one_out)
        #print(one_out_tra)
    print(out_put_tra)
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        out_put_tra_val = sess.run([out_put_tra])
        print(out_put_tra_val)
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return out_put_tra_val, out_put_tra



#numpy.set_printoptions(threshold='nan')

filenames_000001 = ["data/sh_hq_000001_2011.csv", "data/sh_hq_000001_2012.csv", "data/sh_hq_000001_2013.csv", "data/sh_hq_000001_2014.csv", \
    "data/sh_hq_000001_2015.csv", "data/sh_hq_000001_2016.csv"]
filenames_399001 = ["data/sz_hq_399001_2011.csv", "data/sz_hq_399001_2012.csv", "data/sz_hq_399001_2013.csv", "data/sz_hq_399001_2014.csv", \
    "data/sz_hq_399001_2015.csv", "data/sz_hq_399001_2016.csv"]
filenames_399006 = ["data/sz_hq_399006_2011.csv", "data/sz_hq_399006_2012.csv", "data/sz_hq_399006_2013.csv", "data/sz_hq_399006_2014.csv", \
    "data/sz_hq_399006_2015.csv", "data/sz_hq_399006_2016.csv"]
filenames_300188 = ["data/sz_hq_300188_2011.csv", "data/sz_hq_300188_2012.csv", "data/sz_hq_300188_2013.csv", "data/sz_hq_300188_2014.csv", \
    "data/sz_hq_300188_2015.csv", "data/sz_hq_300188_2016.csv"]

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
print(input_val)

real_input_val, real_input_data = get_in_data_last5dat(input_data, 0, input_size - 5)
print(real_input_data)

print("=======================")
output_data_val, output_data = get_out_data_last5day(out_put_300188, output_size)
print("=======================")
#print(out_val_300188)

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


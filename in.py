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

def get_file_line(filename):
    myfile = open(filename) 
    lines = len(myfile.readlines()) 
    return lines

def read(filenames, batch_size):
    # 生成一个先入先出队列和一个QueueRunner,生成文件名队列  
		
    filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
    print(filename_queue.queue_ref)
    lines = get_file_line(filename="data/sh_hq_000001_2011.csv")
    print(lines)
    
    # 定义Reader  
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)  
    # 定义Decoder  
    record_defaults = [['null'], ['null'], ['null'], ['null'], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13 = tf.decode_csv(value,record_defaults=record_defaults)
    features = tf.stack([col3, col1, col2, col4])
    output = tf.stack([col5, col6, col7, col8, col9, col10, col11, col12, col13]) 
    tf.set_random_seed(0)
    example_batch, label_batch = tf.train.batch([features,output], batch_size=batch_size, capacity=2000, num_threads=1)
    print(example_batch, label_batch)
    # 运行Graph  
    with tf.Session() as sess:  
        coord = tf.train.Coordinator()  #创建一个协调器，管理线程  
        threads = tf.train.start_queue_runners(coord=coord)  #启动QueueRunner, 此时文件名队列已经进队。  
        e_val,l_val = sess.run([example_batch, label_batch])  
        #print e_val,l_val  
        coord.request_stop()  
        coord.join(threads)
    return e_val,l_val


filenames = ["data/sh_hq_000001_2011.csv", "data/sh_hq_000001_2012.csv"]
val1, val2 = read(filenames, batch_size=3)
print(val1, val2)

numpy.set_printoptions(threshold='nan')
filenames = ["data/sz_hq_300188_2011.csv", "data/sz_hq_300188_2012.csv"]
val1_300188, val2_300188 = read(filenames, batch_size=2)
print("%s" %val1_300188)
print(val2_300188)
#print("%.3f" %val2_300188)

a=[1.1,2.2,3.3]
print(a)
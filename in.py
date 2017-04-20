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
# ����һ�������ȳ����к�һ��QueueRunner,�����ļ�������
filenames = ['1.csv']
filename_queue = tf.train.string_input_producer(filenames, shuffle=False)
# ����Reader
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)
# ����Decoder
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(value,record_defaults=record_defaults)
features = tf.stack([col1, col2, col3], 0)
label = tf.stack([col4,col5], 0)
example_batch, label_batch = tf.train.shuffle_batch([features,label], batch_size=2, capacity=200, min_after_dequeue=100, num_threads=2)
# ����Graph
with tf.Session() as sess:
    coord = tf.train.Coordinator()  #����һ��Э�����������߳�
    threads = tf.train.start_queue_runners(coord=coord)  #����QueueRunner, ��ʱ�ļ��������Ѿ����ӡ�
    for i in range(10):
        e_val,l_val = sess.run([example_batch, label_batch])
        print(e_val,l_val)
    coord.request_stop()
    coord.join(threads)

'''
import tensorflow as tf
import os

filename = os.path.join("data","contact")
filename_queue = tf.train.string_input_producer(["data/sh_hq_000001_2011.csv", 
                                                 "data/sh_hq_000001_2012.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

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

  for i in range(700):
    # Retrieve a single instance:
    example, label = sess.run([features, D_output])
    #print(example.eval(),label.eval())
    print(example,label)

  coord.request_stop()
  coord.join(threads)
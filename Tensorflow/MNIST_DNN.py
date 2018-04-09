import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#모델구성#
#앞이 None인건 한번에 학습시킬 MNIST 이미지의 개수, 즉 배치 크기 지정
#None으로 넣어주면 텐서플로우가 알아서 계산함
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

# logits = model의 cost를 계산하는듯
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y
))

# cost를 제일 낮게
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 학습단계
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' %(epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('end!')

# 결과 확인
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images,
                                  Y: mnist.test.labels}))

'''
출력 값
Epoch: 0001 Avg. cost =  0.400
Epoch: 0002 Avg. cost =  0.147
Epoch: 0003 Avg. cost =  0.097
Epoch: 0004 Avg. cost =  0.071
Epoch: 0005 Avg. cost =  0.054
Epoch: 0006 Avg. cost =  0.042
Epoch: 0007 Avg. cost =  0.033
Epoch: 0008 Avg. cost =  0.026
Epoch: 0009 Avg. cost =  0.023
Epoch: 0010 Avg. cost =  0.017
Epoch: 0011 Avg. cost =  0.014
Epoch: 0012 Avg. cost =  0.016
Epoch: 0013 Avg. cost =  0.014
Epoch: 0014 Avg. cost =  0.010
Epoch: 0015 Avg. cost =  0.012
end!
정확도: 0.9792
'''
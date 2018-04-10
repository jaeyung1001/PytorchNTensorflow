import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# 모델
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

# dropout 에서 사용하는 변수
# dropout은 DNN에서 과적합(Overfitting)을 방지할 수가 있음
# ex: keep_prob가 0.8일시엔 해당 레이어 노드를 80%만 사용하도록 설정하는 것
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu((tf.matmul(L1, W2)))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 모델 학습ㄱㄱ
init = tf.global_variables_initializer() #멤버변수 초기화
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})
        # keep_prob는 중간(히든)레이어에는 퍼센테이지로 쓸수있지만
        # 마지막 레이어는 모든 노드를 사용해야하므로 무조건 1로 써야함
        total_cost = total_cost + cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print("End! optimizing")

# 결과 확인 ㄱㄱ
# argmax함수는 배열중 최댓값 인덱스를 뽑아내는 함수
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images,
                                  Y: mnist.test.labels,
                                  keep_prob: 1}))   # 결과 확인은 keep_prob은 1로 해줘야함

'''
결과 값:
Epoch: 0001 Avg. cost = 0.430
Epoch: 0002 Avg. cost = 0.164
Epoch: 0003 Avg. cost = 0.113
Epoch: 0004 Avg. cost = 0.088
Epoch: 0005 Avg. cost = 0.071
Epoch: 0006 Avg. cost = 0.063
Epoch: 0007 Avg. cost = 0.051
Epoch: 0008 Avg. cost = 0.045
Epoch: 0009 Avg. cost = 0.040
Epoch: 0010 Avg. cost = 0.039
Epoch: 0011 Avg. cost = 0.034
Epoch: 0012 Avg. cost = 0.032
Epoch: 0013 Avg. cost = 0.030
Epoch: 0014 Avg. cost = 0.026
Epoch: 0015 Avg. cost = 0.026
Epoch: 0016 Avg. cost = 0.025
Epoch: 0017 Avg. cost = 0.023
Epoch: 0018 Avg. cost = 0.022
Epoch: 0019 Avg. cost = 0.022
Epoch: 0020 Avg. cost = 0.021
Epoch: 0021 Avg. cost = 0.020
Epoch: 0022 Avg. cost = 0.020
Epoch: 0023 Avg. cost = 0.019
Epoch: 0024 Avg. cost = 0.017
Epoch: 0025 Avg. cost = 0.019
Epoch: 0026 Avg. cost = 0.017
Epoch: 0027 Avg. cost = 0.018
Epoch: 0028 Avg. cost = 0.016
Epoch: 0029 Avg. cost = 0.015
Epoch: 0030 Avg. cost = 0.016
End! optimizing
정확도: 0.9798
'''
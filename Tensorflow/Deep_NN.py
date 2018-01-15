import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

# [기타, 포유류, 조류]
# one-hot형식
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 현재 만드는 뉴럴 네트워크 모델은 모든 노드가 연결되어있으므로
# 이를 FC(Fully-Connected)라고 표현

# 첫번째 가중치의 차원 [인풋, 히든레이어의 뉴런 갯수]
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
# 두번째 가중치의 차원을 [첫번째 히든레이어의 뉴런 갯수, 분류 갯수]
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.))

# 편향을 각각 각레이어의 아웃풋 갯수로 설정
# b1 은 히든 레이어의 뉴런 갯수로, b2는 최종 결과값 즉, 분류 갯수인 3으로 설정
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 신경망의 히든 레이어에 가중치 W1과 편향 b1을 적용
# X, W1을 곱하는 이유는 행렬곱*때문 그리고 편향 플러스
# | w(1,1) w(2,1)   ( input 1)   | (input_1 * w(1,1)) + (input_2 * w(2,1))|
# | w(1,2) w(2,2) * ( input 2) = | (input_1 * w(1,2)) + (input_2 * w(2,2))|
# 이런식으로 적용
L1 = tf.add(tf.matmul(X,W1), b1)
L1 = tf.nn.relu(L1)

# 출력층 설정
model = tf.add(tf.matmul(L1, W2), b2)

# 손실함수 설정
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model)
)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # 변수들 초기화 해주고

for epoch in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if(epoch + 1) % 10 == 0:
        print(epoch +1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 결과 확인
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("predict: ", sess.run(prediction, feed_dict={X: x_data}))
print("real: ", sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

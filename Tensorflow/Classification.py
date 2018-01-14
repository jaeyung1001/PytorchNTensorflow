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

# 신경망 모델 구성 중요...

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망은 2차원으로 [2, 3]은 2: 입력 노드 수, 3: 출력 노드수
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))

# 편향은 출력층의 갯수, 최종 결과값의 분류 갯수인 3으로 설정
b = tf.Variable(tf.zeros([3]))

# 신경망에 W와 b을 적용
L = tf.add(tf.matmul(X, W), b)

# activate function 적용
L = tf.nn.relu(L)

# 마지막으로 softmax 함수 적용 ==> 출력값을 사용하기 쉽게 만듬
model = tf.nn.softmax(L)

# loss 함수 작성
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis = 1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 신경망 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(1000):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    if (epoch + 1) % 10 == 0:
        print(epoch + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# 결과
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print("예측값: ", sess.run(prediction, feed_dict={X: x_data}))
print("실제값: ", sess.run(target, feed_dict={Y: y_data}))

# 결과 값 검사 equal로 tensor비교
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))
import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

print(X)
print(Y)

hypothesis = W * X + b
# mean(hypothesis - Y)^2 예측값과 실제값의 거리를 손실 함수로 정합니다
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 경사 하강법 최적화 수행
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 손실을 최소화 하는 것이 최종 목표
train_op = optimizer.minimize(cost)

# 세션 생성 및 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        # run에 optimizer는 안넣는 이유는 train_op가 이미 포함되서 인가..?
        # optimizer를 추가하면 에러나넹
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data,
                                                            Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
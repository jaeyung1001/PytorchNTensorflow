import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font_name = matplotlib.font_manager.FontProperties(
    fname="C:/Windows/Fonts/HANBatang.ttf"
).get_name()

matplotlib.rc('font', family=font_name)

# 단어 벡터를 분석해볼 임의의 문장들
sentences = ["나 고양이 좋다",
             "나 강아지 좋다",
             "나 동물 좋다",
             "강아지 고양이 동물",
             "여자친구 고양이 강아지 좋다",
             "고양이 생선 우유 좋다",
             "강아지 생선 싫다 우유 좋다",
             "강아지 고양이 눈 좋다",
             "나 여자친구 좋다",
             "여자친구 나 싫다",
             "여자친구 나 영화 책 음악 좋다",
             "나 게임 만화 애니 좋다",
             "고양이 강아지 싫다",
             "강아지 고양이 좋다"]

word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# word_list 다음과 같음
# ['생선', '음악', '애니', '고양이', '싫다', '게임',
# '나', '우유', '좋다', '영화', '책', '강아지', '눈', '동물', '만화', '여자친구']
# print(word_list)

word_dict = {w: i for i, w in enumerate(word_list)}
# print(word_dict)
# word_dict 다음과 같음
# {'고양이': 0, '눈': 11, '책': 1, '영화': 2, '생선': 3,
# '애니': 4, '게임': 5, '동물': 6, '나': 7, '우유': 9,
# '만화': 12, '싫다': 10, '좋다': 13, '강아지': 14, '음악': 15, '여자친구': 8}
word_index = [word_dict[word] for word in word_list]
# print(word_index)
# word_index는 다음과 같음
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]


# 윈도우 사이즈를 1 로 하는 skip-gram 모델을 만든다.
# 예) 나 게임 만화 애니 좋다
#   -> ([나, 만화], 게임), ([게임, 애니], 만화), ([만화, 좋다], 애니)
#   -> (게임, 나), (게임, 만화), (만화, 게임), (만화, 애니), (애니, 만화), (애니, 좋다)
skip_grams = []

for i in range(1, len(word_index) - 1):
    target = word_index[i]
    context = [word_index[i - 1], word_index[i + 1]]

    for w in context:
        skip_grams.append([target, w])

# print(skip_grams)
# [[1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4],
# [4, 3], [4, 5], [5, 4], [5, 6], [6, 5], [6, 7],
# [7, 6], [7, 8], [8, 7], [8, 9], [9, 8], [9, 10],
# [10, 9], [10, 11], [11, 10], [11, 12], [12, 11],
# [12, 13], [13, 12], [13, 14], [14, 13], [14, 15]]


# skip-gram 데이터에서 무작위로 데이터를 뽑아 입력값과 출력값의 배치 데이터를
# 생성하는 함수
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0]) # target
        random_labels.append([data[i][1]]) # context word

    return random_inputs, random_labels

# 옵션설정
# 학습 반복 수
training_epoch = 300
# 학습률
learning_rate = 0.1
#한 번에 학습할 데이터의 크기
batch_size = 20
# 단어 벡터를 구성할 임베딩 차원의 크기
embedding_size = 2
# word2vec 모델을 학습시키기 위한 nce_loss함수에서 사용하기 위한 샘플링 크기
# batch_size보다 작아야함
num_sampled = 15
# 총 단어 갯수
voc_size = len(word_list)

# 신경망 모델 구성
inputs = tf.placeholder(tf.int32, shape=[batch_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

# word2vec 모델의 결과 값인 임베딩 벡터를 저장할 변수입니다.
# 총 단어 갯수와 임베딩 갯수를 크기로 하는 두 개의 차원을 갖습니다.
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
# 임베딩 벡터의 차원에서 학습할 입력값에 대한 행들을 뽑아옵니다.
# 예) embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]

selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

# nce_loss 함수에서 사용할 변수들을 정의합니다.
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# nce_loss 함수를 직접 구현하려면 매우 복잡하지만,
# 함수를 텐서플로우가 제공하므로 그냥 tf.nn.nce_loss 함수를 사용하기만 하면 됩니다.
loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, labels, selected_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#########
# 신경망 모델 학습
######
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print("loss at step ", step, ": ", loss_val)

    # matplot 으로 출력하여 시각적으로 확인해보기 위해
    # 임베딩 벡터의 결과 값을 계산하여 저장합니다.
    # with 구문 안에서는 sess.run 대신 간단히 eval() 함수를 사용할 수 있습니다.
    trained_embeddings = embeddings.eval()


#########
# 임베딩된 Word2Vec 결과 확인
# 결과는 해당 단어들이 얼마나 다른 단어와 인접해 있는지를 보여줍니다.
######
for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')

plt.show()
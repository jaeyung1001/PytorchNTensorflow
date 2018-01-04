import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad=True)
# Any random value


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# Before training
print("predict (before training)", 4, forward(4).data[0])

# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        # print("\tl: ", l)
        l.backward()
        # out.backward(): x가 포함된 연산의 결과가 out이라고 할 때
        # out을 backpropagation해서 x로 out을 미분한 gradient를 구한다
        # print("\tla: ", l)
        print("\tgrad: ", x_val, y_val, w.grad.data[0], l.data[0])
        w.data = w.data - 0.01 * w.grad.data

        # Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progrss:", epoch, l.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])

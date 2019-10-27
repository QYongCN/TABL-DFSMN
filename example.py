import layer
import torch as tf
import torch.nn as nn
import torch.optim as optim
import keras
import numpy as np

gpus = [0]
cuda_gpu = tf.cuda.is_available()

# 1 hidden layer network with input: 40x10, hidden 120x5, output 3x1
template = [[40,10], [120,5], [3,1]]

# random data
x = np.random.rand(1000,40,10)
y = keras.utils.to_categorical(np.random.randint(0,3,(1000,)),3)
if cuda_gpu:
    x = tf.from_numpy(x).cuda()
    y = tf.from_numpy(y).cuda()
else:
    x = tf.from_numpy(x)
    y = tf.from_numpy(y)

epochs = 1000

model_TABLDFSMN = layer.TABL_DFSMN(template[-1], template[0])
if cuda_gpu:
    model_tabl = nn.DataParallel(model_TABLDFSMN, device_ids=gpus).cuda()


criterion = nn.CrossEntropyLoss()
criterion.cuda()
optimizer = optim.Adam(model_TABLDFSMN.parameters(), lr=0.02)

for epoch in range(epochs):
    print("epoch======>", epoch)
    out = model_TABLDFSMN(x.float())
    loss = criterion(out, tf.max(y, 1)[1])
    print(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



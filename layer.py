import torch as tf
import torch.nn as nn

class TABL_DFSMN(nn.Module):
    def __init__(self, output_dim, input_dim):
        super(TABL_DFSMN, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.W1 = nn.Parameter(tf.Tensor(self.output_dim[0], self.input_dim[0]))
        self.DFSMNW1 = nn.Parameter(tf.Tensor(self.input_dim[1], self.input_dim[1]))
        self.DFSMNW2 = nn.Parameter(tf.Tensor(self.input_dim[1], self.input_dim[1]))
        self.DFSMNW3 = nn.Parameter(tf.Tensor(self.input_dim[1], self.input_dim[1]))
        self.DFSMNW4 = nn.Parameter(tf.Tensor(self.input_dim[1], self.input_dim[1]))
        self.W2 = nn.Parameter(tf.Tensor(self.input_dim[1], self.output_dim[1]))
        self.B = nn.Parameter(tf.Tensor(output_dim[0], output_dim[1]))
        self.softmax = nn.Softmax()

    def forward(self, X):
        X_ = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            X_[i] = self.W1 @ X[i]

        E1 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            E1[i] = X_[i] @ self.DFSMNW1
        Attention1 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        Attention1 = self.softmax(E1)
        M1 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            M1[i] = X_[i] * Attention1[i]
        DFSMN_X1 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            DFSMN_X1[i] = X_[i] + M1[i]

        E2 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            E2[i] = DFSMN_X1[i] @ self.DFSMNW2
        Attention2 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        Attention2 = self.softmax(E2)
        M2 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            M2[i] = DFSMN_X1[i] * Attention2[i] + M1[i]
        DFSMN_X2 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            DFSMN_X2[i] = DFSMN_X1[i] + M2[i]

        E3 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            E3[i] = DFSMN_X2[i] @ self.DFSMNW3
        Attention3 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        Attention3 = self.softmax(E3)
        M3 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            M3[i] = DFSMN_X2[i] * Attention3[i] + M2[i]
        DFSMN_X3 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            DFSMN_X3[i] = DFSMN_X2[i] + M3[i]

        E4 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            E4[i] = DFSMN_X3[i] @ self.DFSMNW4
        Attention4 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        Attention4 = self.softmax(E4)
        M4 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            M4[i] = DFSMN_X3[i] * Attention4[i] + M3[i]
        DFSMN_X4 = tf.Tensor(X.shape[0], self.output_dim[0], self.input_dim[1]).cuda()
        for i in range(X.shape[0]):
            DFSMN_X4[i] = DFSMN_X3[i] + M4[i]

        _X = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1]).cuda()
        for i in range(X.shape[0]):
            _X[i] = DFSMN_X4[i] @ self.W2

        Y = tf.Tensor(X.shape[0], self.output_dim[0], self.output_dim[1])
        for i in range(X.shape[0]):
            Y[i] = _X[i] + self.B

        out = tf.Tensor(X.shape[0], self.output_dim[0]).cuda()
        if self.output_dim[1] == 1:
            for i in range(0, X.shape[0]):
                out[i] = tf.squeeze(Y[i], -1)
            return out

        return Y






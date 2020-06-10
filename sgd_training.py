######################################
# sgd_training.py
# training utils for evaluating pytorch neural networks
# trains models with gradient descent
# also evaluates weights from SMT model
######################################
import torch
import torch.nn.functional as F
from models import OneLayerDNN, TwoLayerDNN


def train_dnn_gd(x, y, xtest, ytest, epochs):
    model = TwoLayerDNN(x.shape[1], 2)
    optimizer = torch.optim.Adam(model.parameters(), 1e-2)
    for epoch in range(epochs):
        output = model(x)
        loss = F.cross_entropy(output, y.long())
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.float().eq(y.float().view_as(pred)).sum().item()

    output_test = model(xtest)
    pred_test = output.argmax(dim=1, keepdim=True)
    correct_test = pred_test.float().eq(ytest.float().view_as(pred_test)).sum().item()
    return correct, loss, correct_test


def eval_smt_dnn2(x, w, b, y, layers=2):
    w1, w2 = w
    b1, b2 = b
    w1 = w1.unsqueeze(0)
    w2 = w2.view(1, 1)
    b1 = b1.unsqueeze(0)
    b2 = b2.unsqueeze(0)

    y1 = F.linear(x, w1, bias=b1)
    y_hat = F.linear(y1, w2, bias=b2)
    return y, y_hat


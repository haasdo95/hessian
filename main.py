import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from enum import Enum


class Mode(Enum):
    NoHessian = 0
    FullHessian = 1
    DiagHessian = 2


mode = Mode.DiagHessian


torch.manual_seed(666)
device = torch.device("cuda")


input_width = 28 * 28  # 784
layer_1_out_width = 20
layer_2_out_width = 10


def fc2_fwd(fc):
    output = fc.reshape(layer_2_out_width, layer_1_out_width) @ fc2_inp
    return F.cross_entropy(output.T, target)
    # output = F.log_softmax(output, dim=0)
    # return F.nll_loss(output.T, target)


if __name__ == '__main__':
    first_order_lr = 0.01
    second_order_lr = first_order_lr
    hessian_boost = 0.01
    epochs = 10
    log_interval = 1
    train_kwargs = {'batch_size': 1000}
    test_kwargs = {'batch_size': 1000}
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    fc1 = torch.zeros((layer_1_out_width, input_width))
    nn.init.xavier_uniform_(fc1, gain=nn.init.calculate_gain('relu'))
    fc1 = fc1.to(device)
    fc1.requires_grad_()

    fc2 = torch.zeros(layer_1_out_width * layer_2_out_width)
    nn.init.uniform_(fc2)
    fc2 = fc2.to(device)
    fc2.requires_grad_()

    for epoch in range(1, epochs + 1):
        # training
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            data = torch.flatten(data, start_dim=1).T  # 784 * batch_size
            fc2_inp = torch.relu(fc1 @ data)
            loss = fc2_fwd(fc2)
            loss.backward()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            with torch.no_grad():
                # with torch.autograd.set_detect_anomaly(True):
                fc1 -= first_order_lr * fc1.grad
                fc1.grad = None
                fc2_grad = fc2.grad
                fc2.grad = None
                if mode == Mode.FullHessian:
                    hessian = torch.autograd.functional.hessian(fc2_fwd, (fc2,), strict=True)[0][0]
                    hessian = hessian.to(device)
                    hessian.diagonal().copy_(hessian.diagonal() + hessian_boost)  # boost diagonal for stability
                    fc2 -= second_order_lr * torch.linalg.lstsq(hessian, fc2_grad).solution  # pseudo inverse
                elif mode == Mode.DiagHessian:
                    hessian = torch.autograd.functional.hessian(fc2_fwd, (fc2,), strict=True)[0][0]
                    hessian = hessian.to(device)
                    diag_inv_hessian = 1 / (hessian.diagonal() + hessian_boost)
                    fc2 -= second_order_lr * diag_inv_hessian * fc2_grad
                else:
                    assert mode == Mode.NoHessian
                    fc2 -= first_order_lr * fc2_grad
        # testing
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                data = torch.flatten(data, start_dim=1).T
                output = torch.relu(fc1 @ data)
                output = fc2.reshape(layer_2_out_width, layer_1_out_width) @ output
                output = F.log_softmax(output, dim=0)
                test_loss += F.nll_loss(output.T, target, reduction="sum").item()
                pred = output.argmax(dim=0, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

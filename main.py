import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(666)
device = torch.device("cuda")

input_width = 28 * 28  # 784
layer_1_out_width = 32
layer_2_out_width = 10


def fc2_fwd(fc):
    output = fc.reshape(layer_2_out_width, layer_1_out_width) @ fc2_inp
    output = F.log_softmax(output, dim=0)
    return F.nll_loss(output.T, target)


if __name__ == '__main__':
    fc1_lr = 0.1
    fc2_lr = 0.1
    epochs = 14
    log_interval = 10
    train_kwargs = {'batch_size': 64}
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
            assert fc1.requires_grad and fc2.requires_grad
            with torch.no_grad():
                fc1 -= fc1_lr * fc1.grad
                fc2 -= fc2_lr * fc2.grad
                fc1.grad = None
                fc2.grad = None
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

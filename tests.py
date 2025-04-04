import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torchvision

import matplotlib.pyplot as plt

from constants import MNIST_LAYERS, MNIST_END, MNIST_GATE
from TreeModel import TreeModel
from SimpleModel import SimpleModel
from ExpertsModel import ExpertsModel

def print_results(name, results):
    print(f'\n \nTesting {name} dataset. \n ------------------------------------------------------------------------ \n')
    headers = ["Metric"] + list(results.keys())
    rows = [
        ["Test Accuracy (%)"] + [f"{results[model]['test_acc']:.2f}" for model in results],
    ]

    # Print the header
    print("{:<20}".format(headers[0]), end="")
    for header in headers[1:]:
        print("{:<15}".format(header), end="")
    print()

    # Print the rows
    for row in rows:
        print("{:<20}".format(row[0]), end="")
        for value in row[1:]:
            print("{:<15}".format(value), end="")
        print()

    # Plot the accuracies and losses
    plt.figure(figsize=(10, 5))

    # Plot losses for all models
    for model_name, result in results.items():
        plt.plot(result['losses'], label=f'{model_name} Loss')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Loss')
    plt.title('Loss Over Time for All Models')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    # Plot accuracies for all models
    for model_name, result in results.items():
        plt.plot(result['accuracies'], label=f'{model_name} Accuracy')
    plt.xlabel('Batch (x100)')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Time for All Models')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def full_mnist(optimizer, loss_fn, epochs=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    models = ['tree', 'simple', 'experts']
    results = {}
    for model_name in models:
        print(f'Running {model_name} model')
        
        model = None
        if model_name == 'tree':
            model = TreeModel(MNIST_LAYERS, MNIST_END, MNIST_GATE)
        elif model_name == 'simple':
            model = SimpleModel(MNIST_LAYERS, MNIST_END)
        elif model_name == 'experts':
            model = ExpertsModel(MNIST_LAYERS, MNIST_END, MNIST_GATE)
        model.to(device)
        temp_optimizer = optimizer(model.parameters())
        results[model_name] = mnist_train_test(model, temp_optimizer, loss_fn, trainloader, testloader, device, epochs=epochs)
    
    print_results('MNIST', results)


def mnist_train_test(model, optimizer, loss_fn, trainloader, testloader, device, epochs=5):
    losses = []
    accuracies = []
    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                correct = 0
                total = 0
                with torch.no_grad():
                    for test_data in testloader:
                        test_images, test_labels = test_data
                        test_images, test_labels = test_images.to(device), test_labels.to(device)
                        test_outputs = model(test_images)
                        _, test_predicted = torch.max(test_outputs.data, 1)
                        total += test_labels.size(0)
                        correct += (test_predicted == test_labels).sum().item()
                losses.append(running_loss / 100)
                accuracies.append(100 * correct / total)

                running_loss = 0.0

    train_time = time.time()-start

    # Test the network on the test data
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    test_time = time.time()-start

    return {'losses': losses, 'accuracies': accuracies, 'test_acc': correct / total}


def full_cifar100(optimizer, loss_fn, epochs=5):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Import constants for CIFAR100
    from constants import CIFAR100_LAYERS, CIFAR100_END, CIFAR100_GATE
    
    models = ['tree', 'simple', 'experts']
    results = {}
    for model_name in models:
        print(f'Running {model_name} model')
        
        model = None
        if model_name == 'tree':
            model = TreeModel(CIFAR100_LAYERS, CIFAR100_END, CIFAR100_GATE)
        elif model_name == 'simple':
            model = SimpleModel(CIFAR100_LAYERS, CIFAR100_END)
        elif model_name == 'experts':
            model = ExpertsModel(CIFAR100_LAYERS, CIFAR100_END, CIFAR100_GATE)
        model.to(device)
        temp_optimizer = optimizer(model.parameters())
        results[model_name] = cifar100_train_test(model, temp_optimizer, loss_fn, trainloader, testloader, device, epochs=epochs)
    
    print_results('CIFAR-100', results)

def cifar100_train_test(model, optimizer, loss_fn, trainloader, testloader, device, epochs=5):
    losses = []
    accuracies = []
    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                correct = 0
                total = 0
                with torch.no_grad():
                    # Use a subset of test data for quick validation during training
                    for j, test_data in enumerate(testloader):
                        if j > 10:  # Limit validation to speed up training
                            break
                        test_images, test_labels = test_data
                        test_images, test_labels = test_images.to(device), test_labels.to(device)
                        test_outputs = model(test_images)
                        _, test_predicted = torch.max(test_outputs.data, 1)
                        total += test_labels.size(0)
                        correct += (test_predicted == test_labels).sum().item()
                losses.append(running_loss / 100)
                accuracies.append(100 * correct / total)
                
                print(f'Epoch {epoch+1}, Batch {i+1}: Loss {running_loss/100:.3f}, Accuracy {100 * correct / total:.2f}%')
                running_loss = 0.0

    train_time = time.time()-start

    # Test the network on the test data
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_time = time.time()-start

    return {'losses': losses, 'accuracies': accuracies, 'test_acc': correct / total}



import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import mlflow
from filelock import FileLock
from torch.utils.data import random_split
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.integration.mlflow import MLflowLoggerCallback
from mlflow.tracking import MlflowClient


def load_data(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform)

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform)

    return trainset, testset

def train_cifar(config):
    # initialize model
    if config['arch'] == 'ResNet34':
        net = models.resnet34(pretrained=config['pretrained'])    
    elif config['arch'] == 'ResNet50':
        net = models.resnet50(pretrained=config['pretrained'])
    elif config['arch'] == 'ResNet101':
        net = models.resnet101(pretrained=config['pretrained'])
    elif config['arch'] == 'ResNet152':
        net = models.resnet101(pretrained=config['pretrained'])        
    else:
        raise ValueError('Invalid architecture provided')

    # modify fully connected output layer
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, 10) # 10 classes in CIFAR dataset

    device = torch.device("cuda")
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # Set optimizers
    if config['opt']=='sgd':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'])
    elif config['opt']=='adam':
        optimizer = optim.Adam(net.parameters(), lr=config['lr'])
    else:
        raise ValueError('Invalid optimizer provided')

    data_dir = os.path.abspath("./data")
    trainset, testset = load_data(data_dir)

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])
    
    train_data_size = len(train_subset)

    trainloader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)
    valloader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8)

    epochs = 5
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        train_loss = 0.0
        train_acc = 0.0
        print("Epoch: {}/{}".format(epoch+1, epochs))
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)            

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        train_acc = train_acc / train_data_size
        train_loss = train_loss / train_data_size        
        
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        tune.report(train_loss=train_loss, train_accuracy=train_acc, val_loss=(val_loss / val_steps), val_accuracy=correct / total)
    
    # test model on test set
    _, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2)

    best_trained_model = net
    
    correct = 0
    total = 0
    with torch.no_grad():
        best_trained_model.eval()
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 

    test_accuracy = float(correct / total)
    tune.report(test_accuracy=test_accuracy)   
    
    # Save weights
    out_name = ''
    for k,v in config.items():
        if not k in ['weights_dir','cwd']:
            out_name += '{}-{}_'.format(k,v)    
    torch.save(best_trained_model.state_dict(),os.path.join(config['cwd'],config['weights_dir'],out_name[:-1]+'.pth'))
    print("Finished Training")          

if __name__=="__main__":
    # Initialize MLflow client
    client = MlflowClient()

    cwd = os.getcwd()

    exp_base_name = sys.argv[1]

    # Create new MLflow experiment with name specified in command line arg
    # 5 max retries renaming exp name
    created = 0
    for i in range(100):
        try:
            exp_name = exp_base_name+"_{}".format(i)
            experiment_id = client.create_experiment(exp_name)
            created=1
            break
        except (TypeError,mlflow.exceptions.MlflowException):
            continue

    if not created:
        print("ERROR: Try new experiment name.")
        sys.exit(1)

    # Copy script to corresponding MLflow experiment dir
    shutil.copyfile(__file__,'mlruns/{}/{}'.format(experiment_id,__file__))

    # Specify where weights will be stored out
    weights_root = './model_weights/'
    weights_dir = weights_root+exp_name+'/'
    os.mkdir(weights_dir)

    # Run hyperparameter search, method used based on command line arg
    if sys.argv[2]=='basic_search':
        # "basic search": consists of grid search, random sampling, etc...

        """
        tune.run(...) will perform the hyperparameter sweep
        The first argument is a function that will:
            - load and preprocess training/validation data
            - define the model
            - train the model
            - log desired metrics
            - save out weights if desired
        The config argument will define the search space. The "mlflow_experiment_id"
        will allow params and metrics to be logged to the correct MLflow experiment.
        In the "resources_per_trial" argument, we can specify how many CPUs and GPUs
        we want to provide to each training run within the experiment.
        The "num_samples" argument defines how many training runs will be performed.
        In the "loggers" argument, we specify that we want to log to MLflow.
        """
        config = {
            "mlflow_experiment_id": experiment_id,
            "weights_dir": weights_dir,
            "cwd":cwd,
            "lr": tune.loguniform(1e-4, 1e-1),
            "opt":tune.grid_search(['sgd', 'adam']),
            "arch":tune.grid_search(['ResNet34', 'ResNet50', 'ResNet101']),
            "pretrained":tune.grid_search(['True', 'False']),
            "batch_size": tune.choice([2, 4, 8, 16]),
        }                    
        analysis = tune.run(
            train_cifar, 
            config=config,
            resources_per_trial={"cpu": 4, "gpu": 1},
            num_samples=1,
            callbacks=[MLflowLoggerCallback(experiment_name=exp_name)],
        )
    elif sys.argv[2]=='hyperopt_search':
        # "hyperopt search": is a more sophisticated method of hyperparameter optimization
        # More info here: https://github.com/hyperopt/hyperopt
        # Ray Tune provides support for several other popular hyperparameter optimization packages such as this one

        # Define the search space:
        space = {
            'lr': hp.uniform('lr', 0.001, 0.1),
            'opt': hp.choice('opt', ['sgd', 'adam']),
            'arch': hp.choice('arch', ['ResNet50', 'ResNet101'])
            }

        # Set current best params which are used by hyperopt's algorithm
        current_best_params = [{
            'lr': 0.01,
            'opt': 0,
            'arch': 'ResNet50'
            }]

        # Initialize the HyperOptSearch with our search space, target metric, current best params
        # The "mode" argument specifies that we want to maximize this metric
        algo = HyperOptSearch(space, max_concurrent=4, metric="test_acc", mode="max",
                                points_to_evaluate=current_best_params)

        """
        tune.run(...) will perform the hyperparameter sweep
        The first argument is a function that will:
            - load and preprocess training/validation data
            - define the model
            - train the model
            - log desired metrics
            - save out weights if desired
        In the "config" argument we specify the MLflow experiment id to log to
        and the directory to save weights to.
        Pass the HyperOptSearch as the "search_alg" argument.
        In the "resources_per_trial" argument, we can specify how many CPUs and GPUs
        we want to provide to each training run within the experiment.
        The "num_samples" argument defines how many training runs will be performed.
        In the "loggers" argument, we specify that we want to log to MLflow.
        """
        analysis = tune.run(
            train,
            config={"mlflow_experiment_id": experiment_id,"weights_dir": weights_dir,"cwd":cwd},
            search_alg=algo,
            num_samples=10,
            resources_per_trial={"cpu": 4, "gpu": 1},
            callbacks=[MLflowLoggerCallback(experiment_name=exp_name)]
        )
    else:
        print("ERROR: Invalid search type. Options: 'basic_search' or 'hyperopt_search'")
        sys.exit(1)



import numpy
from torch.autograd import backward
import wandb
from tqdm import tqdm
import torch
from loss import calc_accuracy
from sliver_net import data


def train(model, criterion, optimizer, train_loader, validation_loader, epochs, device, title):
    for epoch in tqdm(range(epochs)):
        # Train
        running_train_loss = 0.0
        correct_predictions = {}
        incorrect_predictions = {}
        batches_size = {}

        for key in data.LABELS.keys():
            incorrect_predictions[key] = 0.0
            correct_predictions[key] = 0.0
            batches_size[key] = 0.0

        for idx, (train_images, train_labels) in enumerate(train_loader):
            loss, ip, cp, bs = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                          images=train_images, labels=train_labels, mode='train')

            running_train_loss += loss
            for key, label in data.LABELS.items():
                incorrect_predictions[key] += ip[key]
                correct_predictions[key] += cp[key]
                batches_size[key] += bs[key]

                if idx % 100 == 0:
                    wandb.log({'Train/Accuracy/{label_key}'.format(label_key=key):
                               correct_predictions[key] / batches_size[key],
                               'Train/IOU/{label_key}'.format(label_key=key):
                               correct_predictions[key] / (batches_size[key] + incorrect_predictions[key]),
                               'Train/loss': running_train_loss/(idx+1),
                               'Train/epoch': epoch})

            # # Evaluate
            # if idx % 400 == 0:
            #     avg_iou, avg_accuracy = evaluate(model, validation_loader, 'Evaluation', device=device)
            #     print('EVAL: Average IOU:{}, Accuracy:{}'.format(avg_iou, avg_accuracy))


def evaluate(model, dataset_loader, title, device):
    correct_predictions = {}
    incorrect_predictions = {}
    batches_size = {}
    avg_accuracy = 0.0
    avg_iou = 0.0

    for key in data.LABELS.keys():
        correct_predictions[key] = 0.0
        incorrect_predictions[key] = 0.0
        batches_size[key] = 0.0

    for images, labels in dataset_loader:
        _, ip, cp, bs = train_loop(model=model, criterion=None, optimizer=None, device=device,
                                   images=images, labels=labels, mode='eval')

        for key, label in data.LABELS.items():
            incorrect_predictions[key] += ip[key]
            correct_predictions[key] += cp[key]
            batches_size[key] += bs[key]

    for key in data.LABELS.keys():
        accuracy = correct_predictions[key] / batches_size[key]
        iou = correct_predictions[key] / (batches_size[key] + incorrect_predictions[key])
        wandb.log({'{title}/Accuracy/{label_key}'.format(title=title, label_key=key): accuracy,
                   '{title}/IOU/{label_key}'.format(title=title, label_key=key): iou})

        avg_iou += iou / len(data.LABELS)
        avg_accuracy += accuracy / len(data.LABELS)

    return avg_iou, avg_accuracy


def train_loop(model, criterion, optimizer, device, images, labels, mode):
    # Make sure mode is as expected
    if mode == "train" and not model.training:
        model.train()
    elif mode == "eval" and model.training:
        model.eval()

    # Move to device
    images, labels = images.to(device=device, dtype=torch.float), labels.to(device=device, dtype=torch.float)

    # Run the model on the input batch
    pred_scores = model(images)

    # Calculate the accuracy, per key
    correct_predictions = {}
    incorrect_predictions = {}
    batches_size = {}
    for key, label in data.LABELS.items():
        ip, cp, bs = calc_accuracy(pred_scores, labels, specific_label=label)
        incorrect_predictions[key] = ip
        correct_predictions[key] = cp
        batches_size[key] = bs

    loss_value = 0.0

    if mode == "train":
        loss = criterion(pred_scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

    return loss_value, incorrect_predictions, correct_predictions, batches_size

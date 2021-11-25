from torch.autograd import backward
import wandb
from tqdm import tqdm
import torch
from loss import calc_accuracy


def train(model, criterion, optimizer, train_loader, validation_loader, epochs, device, title):
    for epoch in tqdm(range(epochs)):
        # Train
        running_train_loss, running_train_accuracy = 0.0, 0.0
        for train_images, train_labels in train_loader:
            loss, accuracy = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                        images=train_images, labels=train_labels, mode='train')

            running_train_loss += loss
            running_train_accuracy += accuracy
        wandb.log({'{title}/Train/loss'.format(title=title): running_train_loss/len(train_loader),
                   '{title}/Train/accuracy'.format(title=title): running_train_accuracy/len(train_loader),
                   '{title}/Train/epoch': epoch})

        # Validate
        running_validation_accuracy = 0.0
        for validation_images, validation_labels in validation_loader:
            _, accuracy = train_loop(model=model, criterion=criterion, optimizer=optimizer, device=device,
                                     images=validation_images, labels=validation_labels, mode='eval')
            running_validation_accuracy += accuracy
        wandb.log({'{title}/Validation/accuracy'.format(title=title): running_validation_accuracy/len(validation_loader),
                   '{title}/epoch'.format(title=title): epoch})


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

    # Calculate the loss & accuracy
    accuracy = calc_accuracy(pred_scores, labels, specific_label=None)
    loss_value = 0

    if mode == "train":
        loss = criterion(pred_scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

    return loss_value, accuracy

import cv2
import numpy as np
from data.transforms import transform
from data.functions import get_image
from tqdm import tqdm
import torch
import gc
from matplotlib import pyplot as plt
import os


def train_one_epoch(model, loader, optimizer, loss_fn, epoch_num=-1, device='cpu', plotting=False):
    loop = tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()


    for i, batch in loop:
        colored, gray = batch
        colored = colored.to(device)
        gray = gray.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(gray)

        # loss calculation
        loss = loss_fn(outputs, colored)

        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        if i % 10 == 0:
            gc.collect()

        loop.set_postfix({"loss": float(loss)})

    if plotting:
        return float(loss)
    return


def val_one_epoch(
    model,
    loader,
    loss_fn,
    best_so_far=0.0,
    best=float("inf"),
    ckpt_path="./models/best.pt",
    epoch_num=-1,
    device='cpu',
    plotting=False,
    visual_progress=False,
):

    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )

    with torch.no_grad():
        loss = float("inf")
        model.eval()  # evaluation mode
        for i, batch in loop:
            colored, gray = batch
            colored = colored.to(device)
            gray = gray.to(device)

            # forward pass
            outputs = model(gray)

            # loss calculation
            loss = loss_fn(outputs, colored)

            loop.set_postfix({"mse": float(loss)})

            if i % 10 == 0:
                gc.collect()

        if loss < best_so_far:
            torch.save(model.state_dict(), ckpt_path)
            best_so_far = loss

    if visual_progress and epoch_num % 3 == 0:
        triplet = None
        t1 = gray[0][1].squeeze().cpu()
        t2 = colored[0].permute(1, 2, 0).cpu()
        t3 = outputs[0].permute(1, 2, 0).cpu()
        triplet = (t1, t2, t3)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(triplet[0], cmap='gray')  # Grayscale input
        axes[0].set_title("Input (Grayscale)")

        axes[1].imshow(triplet[1])      # Model output (RGB)
        axes[1].set_title("Truth")

        axes[2].imshow(triplet[2])  # Ground truth (RGB)
        axes[2].set_title("Model Output")

        dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/plots'
        filename = dir + '/visual_progress_epoch_' + str(epoch_num) + '.png'
        plt.savefig(filename)

    return best_so_far, float(loss)


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    epochs=10,
    ckpt_path="./models/best.pt",
    device='cpu',
    plotting=False,
):
    best = float("inf")
    prev_best = best
    counter = 0
    losses_train = []
    losses_val = []
    for epoch in range(epochs):
        if plotting:
            # loss_train = train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch,device=device, plotting=True)
            # losses_train.append(loss_train)
            best, loss_val = val_one_epoch(
                model,
                val_dataloader,
                loss_fn,
                best_so_far=best,
                ckpt_path=ckpt_path,
                epoch_num=epoch,
                device=device,
                plotting=True,
                visual_progress=True,
            )
            losses_val.append(loss_val)

        else:
            train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch, device=device)
            best = val_one_epoch(
                model,
                val_dataloader,
                loss_fn,
                best_so_far=best,
                ckpt_path=ckpt_path,
                device=device,
                epoch_num=epoch,
            )

        if prev_best - best <= 0.0000001:
            counter += 1
        else:
            counter = 0
        if best < prev_best:
            prev_best = best
        if counter >= 5:
            break
    if plotting:
        plt.figure(figsize=(10, 5))
        plt.plot(losses_train, label='Training Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()

        dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/plots'
        filename = dir + '/train_val_loss.png'
        plt.savefig(filename)

import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss

# Define paths
dir_img = Path(r'C:\Users\shore\Desktop\APS360\ISIC2018\seg_data\images')
dir_mask = Path(r'C:\Users\shore\Desktop\APS360\ISIC2018\seg_data\masks')
dir_checkpoint = Path('./checkpoints/')

def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        val_percent=val_percent, save_checkpoint=save_checkpoint,
        img_scale=img_scale, amp=amp
    ))

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate,
                              weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info(f'Validation Dice score: {val_score}')

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--batch-size', '-b', type=int, default=1)
    parser.add_argument('--learning-rate', '-l', type=float, default=1e-5, dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False)
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--validation', '-v', type=float, default=10.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=2)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.validation / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('❌ CUDA OutOfMemoryError! Try reducing image scale (--scale) or enabling AMP (--amp).')
        torch.cuda.empty_cache()
        if hasattr(model, 'use_checkpointing'):
            model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.validation / 100,
            amp=args.amp
        )

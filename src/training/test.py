import gc

import torch

from tqdm import tqdm

from validation import criterion


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, config, optimizer):
    model.eval()

    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, data in bar:
        more_toxic_ids = data["more_toxic_ids"].to(device, dtype=torch.long)
        more_toxic_mask = data["more_toxic_mask"].to(device, dtype=torch.long)
        less_toxic_ids = data["less_toxic_ids"].to(device, dtype=torch.long)
        less_toxic_mask = data["less_toxic_mask"].to(device, dtype=torch.long)
        targets = data["target"].to(device, dtype=torch.long)

        batch_size = more_toxic_ids.size(0)

        more_toxic_outputs = model(more_toxic_ids, more_toxic_mask)
        less_toxic_outputs = model(less_toxic_ids, less_toxic_mask)

        loss = criterion(more_toxic_outputs, less_toxic_outputs, targets, config)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(
            Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]["lr"]
        )

    gc.collect()

    return epoch_loss

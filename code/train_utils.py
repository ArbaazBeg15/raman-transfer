import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import r2_score
from tqdm.auto import tqdm
import neptune


def setup_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_float32_matmul_precision("high")
    
    
def get_scheduler(optimizer, train_dl, epochs):
    total_training_steps = len(train_dl) * epochs
    warmup_steps = int(total_training_steps * 0.05)  # e.g. 5% warmup
    
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps
    )
    
    
def build_loader(
    SEED,
    ds,
    train=True,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    drop_last=True,
    pin_memory=True,
    persistent_workers=False,
):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(SEED if train else SEED+5232)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        #sampler=DistributedSampler(
        #    train_ds,
        #    shuffle=True,
        #    drop_last=True,
        #    seed=config.seed
        #)
    )
    
    
def return_dls(SEED, train_ds, eval_ds, train_batch_size, eval_batch_size):
    train_dl = build_loader(
        SEED,
        train_ds,
        train=True,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
    )

    eval_dl = build_loader(
        SEED,
        eval_ds,
        train=False,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        persistent_workers=False,
    )
    
    return train_dl, eval_dl


def loss_fn(logits, targets):
    logits = logits.view(-1)
    targets = targets.view(-1)
    return F.mse_loss(logits, targets)


def metric_fn(logits, targets):
    preds = logits.cpu().detach().float().numpy()
    targets = targets.cpu().detach().numpy()
    
    dim1 = r2_score(targets[:, 0], preds[:, 0])
    dim2 = r2_score(targets[:, 1], preds[:, 1])
    dim3 = r2_score(targets[:, 2], preds[:, 2])
    
    return dim1, dim2, dim3, r2_score(targets, preds)


class MSEIgnoreNans(_Loss):
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        mask = torch.isfinite(target)
        mse = torch.mean(
            torch.mul(
                torch.square(input[mask] - target[mask]),
                torch.tile(weights[:, None], dims=(1, target.shape[1]))[mask],
            )
        )
        return torch.where(
            torch.isfinite(mse),
            mse,
            torch.tensor(0.).to(target.device),
        )


def setup_neptune(
    seed,
    model_name, 
    lr, 
    weight_decay, 
    epochs,
    batch_size,
    dropout=None,
    drop_path_rate=None,
    resume=False
):
    if not resume:
        neptune_run = neptune.init_run(
            project="arbaaz/kaggle-spect",
            name=model_name,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOGE2YjNiZS1mZGUyLTRjYjItYTg5Yy1mZWJkZTIzNzE1NmIifQ=="
        )

        neptune_run["h_parameters"] = {
            "seed": seed,
            "model_name": model_name,
            "optimizer_name": "nadam",
            "learning_rate": lr,
            "scheduler_name": "default",
            "weight_decay": weight_decay,
            "num_epochs": epochs,
            "batch_size": batch_size,
        }
        
        if dropout: neptune_run["h_parameters"] = {"dropout": dropout}
        if drop_path_rate: neptune_run["h_parameters"] = {"drop_path_rate": drop_path_rate}
        
    else:
        neptune_run = neptune.init_run(
            project="arbaaz/crunchdao-structural-break",
            with_id=config.with_id,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOGE2YjNiZS1mZGUyLTRjYjItYTg5Yy1mZWJkZTIzNzE1NmIifQ=="
        )

    return neptune_run


def train(
    model, 
    optimizer,
    device,
    scaler, 
    scheduler,
    train_dl,
    eval_dl,
    loss_fn,
    epochs,
    checkpoint_name,
    score=-float("inf"),
    neptune_run=None,
    p=True,
):  
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        all_logits = []
        all_targets = []
        
        for inputs, targets, weights in train_dl:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=True):
                logits = model(inputs)
                loss = loss_fn(logits, targets, weights)
            
            loss.backward()
            optimizer.step()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            scheduler.step()
            if neptune_run is not None:  neptune_run["lr_step"].append(scheduler.get_last_lr()[0])
            
            total_loss += loss.detach().cpu()
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
        
        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets)

        one, two, three, r2 = metric_fn(all_logits, all_targets)
        total_loss = total_loss / len(train_dl)
        
        model.eval()
        eval_total_loss = 0.0
        eval_all_logits = []
        eval_all_targets = []

        for inputs, targets, weights in eval_dl:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            with torch.inference_mode():
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, cache_enabled=True):
                    logits = model(inputs)
                    loss = loss_fn(logits, targets, weights)

            eval_total_loss += loss.detach().cpu()
            eval_all_logits.append(logits.detach().cpu())
            eval_all_targets.append(targets.detach().cpu())
        
        eval_all_logits = torch.cat(eval_all_logits)
        eval_all_targets = torch.cat(eval_all_targets)

        eval_one, eval_two, eval_three, eval_r2 = metric_fn(eval_all_logits, eval_all_targets)
        eval_total_loss = eval_total_loss / len(eval_dl)
        
        if eval_r2 > score:
            score = eval_r2
            data = {"state_dict": model.state_dict()}
            data["epoch"] = epoch 
            data["score"] = score
            torch.save(data, f"/checkpoints/{checkpoint_name}")
        
        if neptune_run is not None:
            neptune_run["train/loss"].append(total_loss)
            neptune_run["eval/loss"].append(eval_total_loss)
            neptune_run["train/r2"].append(r2)
            neptune_run["eval/r2"].append(eval_r2)
            #neptune_run["train/one"].append(one)
            #neptune_run["train/two"].append(two)
            #neptune_run["train/three"].append(three)
            #neptune_run["eval/one"].append(eval_one)
            #neptune_run["eval/two"].append(eval_two)
            #neptune_run["eval/three"].append(eval_three)
            
        if p and epoch % 5 == 0:
            tqdm.write(
                f"Epoch: {epoch}, "
                f"train/loss: {total_loss:.4f}, "
                f"eval/loss: {eval_total_loss:.4f}, "
                f"train/r2: {r2:.4f}, "
                f"eval/r2: {eval_r2:.4f}, "
                #f"train/one: {one:.4f}, "
                #f"train/two: {two:.4f}, "
                #f"train/three: {three:.4f}, "
                #f"eval/one: {eval_one:.4f}, "
                #f"eval/two: {eval_two:.4f}, "
                #f"eval/three: {eval_three:.4f} "
            )
            
    if neptune_run is not None: neptune_run.stop()
    return score
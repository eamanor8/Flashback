import random
from pathlib import Path

import numpy as np
import torch
from dataloader import PoiDataloader
from dataset import Split
from evaluation import Evaluation
from network import create_h0_strategy
from setting import Setting
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trainer import FlashbackTrainer

DISABLE_EVALUATION = True  #* enable to save Colab GPU resources if you just want to train
SEED = 42

'''
Main train script to invoke from commandline.
'''

### parse settings ###
setting = Setting()
setting.parse()
print(setting)

###* load previous checkpoints, if present ###
checkpoint_paths = sorted(
    [p for p in Path("saved_models").iterdir() if p.name.startswith("checkpoint_epoch")],
    key=lambda p: (len(p.name), p.name)  #* lexicographical sort that places epoch9 before epoch10
)
checkpoint = torch.load(checkpoint_paths[-1]) if len(checkpoint_paths) > 0 else None

###* reproducible sources of randomness ###
random.seed(SEED)
np.random.seed(SEED)
torch_generator = torch.manual_seed(SEED)

### load dataset ###
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, generator=torch_generator)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, generator=torch_generator)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

### create flashback trainer ###
trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)  #* initial hidden state to RNN
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
if not DISABLE_EVALUATION:
    evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
print('{} {}'.format(trainer, setting.rnn_factory))

###  training loop ###
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

###* resume from previous checkpoints, if present ###
start_epoch_exclusive = -1
if checkpoint is not None:
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    trainer.model.train()
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch_exclusive = checkpoint["epoch"]
    print("Previously left off at...")
    print(f'...Epoch: {start_epoch_exclusive}/{setting.epochs}')
    print(f'...Used learning rate: {scheduler.get_lr()[0]}')
    print(f'...Avg Loss: {checkpoint["epoch_loss"]}')

    ###* reproducible sources of randomness ###
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])


for e in tqdm(
    range(start_epoch_exclusive + 1, setting.epochs),
    desc="epoch",
    total=setting.epochs,
    initial=start_epoch_exclusive + 1
):
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users(random.getstate()) # shuffle users before each epoch!

    losses = []

    for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in enumerate(tqdm(dataloader, desc="train", leave=False)):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)
        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)

        optimizer.zero_grad()
        loss, h = trainer.loss(x, t, s, y, y_t, y_s, h, active_users)
        loss.backward(retain_graph=True)
        losses.append(loss.item())
        optimizer.step()
        #* See https://stackoverflow.com/a/53975741 for a discussion on how loss.backward() and
        #* optimizer.step() are linked: essentially the gradients are stored on the model parameter
        #* tensors themselves; model parameter tensors have an implicit computational graph, and when
        #* loss takes in the predicted label (and compares it with the actual label), gradients are
        #* calculated from predicted label backwards through its computational graph. These gradients
        #* are stored on the tensors involved in the computational graph. optimizer.step() then iterates
        #* through model parameters and uses the stored gradients to update them.

    # schedule learning rate:
    scheduler.step()

    # statistics:
    if (e+1) % 1 == 0:
        epoch_loss = np.mean(losses)
        print(f'Epoch: {e+1}/{setting.epochs}')
        print(f'Used learning rate: {scheduler.get_lr()[0]}')
        print(f'Avg Loss: {epoch_loss}')
    if not DISABLE_EVALUATION:
        if (e+1) % setting.validate_epoch == 0:
            print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
            evaluation_test.evaluate()

    torch.save(
        {
            "epoch": e,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_loss": epoch_loss,
            "random_state": random.getstate(),
            "np_random_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state()
        },
        f"saved_models/checkpoint_epoch{e}.pt"
    )

torch.save(trainer.model.state_dict(), f"saved_models/pretrained.pt")

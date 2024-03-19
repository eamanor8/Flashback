# OK

import random

import numpy as np
import torch
from dataloader import PoiDataloader
from dataset import Split
from evaluation import Evaluation
from network import FixNoiseStrategy, create_h0_strategy
from setting import Setting
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trainer import FlashbackTrainer

SEED = 42

'''
Main train script to invoke from commandline.
'''

### parse settings ###
setting = Setting()
setting.parse()
print(setting)

###* reproducible sources of randomness ###
random.seed(SEED)
np.random.seed(SEED)
torch_generator = torch.manual_seed(SEED)

### load dataset ###
poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
poi_loader.read(setting.dataset_file)
dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

### create flashback trainer ###
trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
assert not setting.is_lstm, f"setting.is_lstm=True is not supported"
h0_strategy: FixNoiseStrategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)  #* initial hidden state to RNN
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)
evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)
print('{} {}'.format(trainer, setting.rnn_factory))

###  training loop ###
optimizer = torch.optim.Adam(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)

for e in tqdm(range(setting.epochs), desc="epoch", total=setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)  #* hidden states are preserved across dataloader iterations for as long as each user's sequence lasts. Shape is (1, setting.batch_size, setting.hidden_dim).
    dataset.shuffle_users() # shuffle users before each epoch!

    losses = []

    #* for shapes of x, t, s, y, y_t, y_s, reset_h, active_users, see bottom of dataset.py
    #* note that until we call squeeze, there is a prepended batch dimension
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

        #* Need to squeeze: dataloader prepends the batch dimension, which is 1
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
    if (e+1) % setting.validate_epoch == 0:
        print(f'~~~ Test Set Evaluation (Epoch: {e+1}) ~~~')
        evaluation_test.evaluate()  #! WARNING: Can take a while if dataset is huge

torch.save(
    {
        "model_state_dict": trainer.model.state_dict(),
        "h0": h0_strategy.h0
    },
    f"saved_models/pretrained.pt"
)

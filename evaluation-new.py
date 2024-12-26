import numpy as np
import torch
from tqdm.auto import tqdm
import time

class Evaluation:
    '''
    Handles evaluation on a given POI dataset and loader.
    The metrics computed per user include total predictions, correct predictions, and accuracy.
    '''

    def __init__(self, dataset, dataloader, user_count, h0_strategy, trainer, setting):
        self.dataset = dataset
        self.dataloader = dataloader
        self.user_count = user_count
        self.h0_strategy = h0_strategy
        self.trainer = trainer
        self.setting = setting
    
    def evaluate(self):
        self.dataset.reset()
        h = self.h0_strategy.on_init(self.setting.batch_size, self.setting.device)

        start_time = time.time()

        # Initialize metrics for each user
        user_total_predictions = np.zeros(self.user_count, dtype=int)
        user_correct_predictions = np.zeros(self.user_count, dtype=int)
        total_users_evaluated = 0
        reset_count = torch.zeros(self.user_count)

        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Evaluating", unit="batch", leave=True)
            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in progress_bar:
                active_users = active_users.squeeze(dim=0)

                for j, reset in enumerate(reset_h):
                    if reset:
                        if self.setting.is_lstm:
                            hc = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                            h[0][0, j] = hc[0]
                            h[1][0, j] = hc[1]
                        else:
                            h[0, j] = self.h0_strategy.on_reset_test(active_users[j], self.setting.device)
                        reset_count[active_users[j]] += 1

                # Preprocess input tensors
                x = x.squeeze(dim=0).to(self.setting.device)
                t = t.squeeze(dim=0).to(self.setting.device)
                s = s.squeeze(dim=0).to(self.setting.device)
                y = y.squeeze(dim=0).to(self.setting.device)
                y_t = y_t.squeeze(dim=0).to(self.setting.device)
                y_s = y_s.squeeze(dim=0).to(self.setting.device)

                active_users = active_users.to(self.setting.device)

                out, h = self.trainer.evaluate(x, t, s, y_t, y_s, h, active_users)
                out = out.permute(1, 0, 2)  # (batch_size, sequence_length, loc_count)
                y = y.permute(1, 0)  # (batch_size, sequence_length)

                for j in range(out.shape[0]):
                    o = out[j].cpu().detach().numpy()
                    ind = np.argpartition(o, -10, axis=1)[:, -10:]  # Top-10 indices
                    y_j = y[j]

                    for k in range(len(y_j)):
                        predictions = torch.tensor(ind[k][np.argsort(-o[k, ind[k]])], device=self.setting.device)  # Sorted top-10 predictions
                        correct = (predictions == y_j[k]).sum().item()

                        # Update user-specific metrics
                        user_id = active_users[j].item()
                        user_total_predictions[user_id] += 1
                        user_correct_predictions[user_id] += correct

            elapsed_time = time.time() - start_time

            # Print results for each user
            for user_id in range(self.user_count):
                if user_total_predictions[user_id] > 0:
                    total_users_evaluated += 1
                    accuracy = user_correct_predictions[user_id] / user_total_predictions[user_id]
                    print(f"User {user_id} - Total Predictions: {user_total_predictions[user_id]}, Correct Predictions: {user_correct_predictions[user_id]}, Accuracy: {accuracy:.2f}")

            print(f"Total Users Evaluated: {total_users_evaluated}")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
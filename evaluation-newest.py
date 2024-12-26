import numpy as np
import torch
from tqdm.auto import tqdm
import time

class Evaluation:
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
        user_metrics = {user_id: {'total_predictions': 0, 'correct_predictions': [0, 0, 0]} for user_id in range(self.user_count)}

        start_time = time.time()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc="Evaluating", unit="batch", leave=True)

            for i, (x, t, s, y, y_t, y_s, reset_h, active_users) in progress_bar:
                # Preprocess input tensors
                x = x.squeeze(dim=0).to(self.setting.device)
                t = t.squeeze(dim=0).to(self.setting.device)
                s = s.squeeze(dim=0).to(self.setting.device)
                y = y.squeeze(dim=0).to(self.setting.device)
                y_t = y_t.squeeze(dim=0).to(self.setting.device)
                y_s = y_s.squeeze(dim=0).to(self.setting.device)

                active_users = active_users.to(self.setting.device)
                # active_users = active_users.squeeze(dim=0).tolist()

                out, h = self.trainer.evaluate(x, t, s, y_t, y_s, h, active_users)
                out = out.permute(1, 0, 2)  # Adjust shape for easy access: (batch_size, sequence_length, loc_count)
                y = y.permute(1, 0)  # (batch_size, sequence_length)

                for j in range(out.shape[0]):  # Iterate over batch size
                    user_id = active_users[j].item()
                    predictions = out[j].cpu().numpy()
                    true_locations = y[j].cpu().numpy()
                    
                    # Calculate top-k indices for recall@k
                    top1_indices = np.argsort(-predictions, axis=1)[:, :1]
                    top5_indices = np.argsort(-predictions, axis=1)[:, :5]
                    top10_indices = np.argsort(-predictions, axis=1)[:, :10]

                    for k, true_loc in enumerate(true_locations):
                        user_id = active_users[j]
                        user_metrics[user_id]['total_predictions'] += 1

                        # Check for recall accuracies
                        user_metrics[user_id]['correct_predictions'][0] += (true_loc in top1_indices[k])
                        user_metrics[user_id]['correct_predictions'][1] += (true_loc in top5_indices[k])
                        user_metrics[user_id]['correct_predictions'][2] += (true_loc in top10_indices[k])

        # Calculate and print Recall@k for each user
        for user_id, metrics in user_metrics.items():
            total_preds = metrics['total_predictions']
            if total_preds > 0:
                recall_at_1 = metrics['correct_predictions'][0] / total_preds
                recall_at_5 = metrics['correct_predictions'][1] / total_preds
                recall_at_10 = metrics['correct_predictions'][2] / total_preds
                print(f"User {user_id} - Recall@1: {recall_at_1:.2f}, Recall@5: {recall_at_5:.2f}, Recall@10: {recall_at_10:.2f}")

        elapsed_time = time.time() - start_time
        print(f"Elapsed Time: {elapsed_time:.2f} seconds")



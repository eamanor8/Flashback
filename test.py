# test.py
import random
import torch
from dataloader import PoiDataloader
from dataset import Split
from evaluation import Evaluation
from network import FixNoiseStrategy, create_h0_strategy
from setting import Setting
from torch.utils.data import DataLoader
from trainer import FlashbackTrainer

SEED = 42

# Main test script to evaluate a pre-trained model
def main():
    # Parse settings
    setting = Setting()
    setting.parse()
    print(setting)

    # Set reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load the dataset
    poi_loader = PoiDataloader(43326, max_users=setting.max_users, min_checkins=setting.min_checkins) # loc_count: 4sq: 68881 gowalla: 121851
    poi_loader.read(setting.dataset_file)

    # Create the test dataset and dataloader with split
    dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Ensure the batch size is valid
    assert setting.batch_size <= poi_loader.user_count(), 'batch size must be lower than the number of available users'

    # Initialize the FlashbackTrainer and model
    trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s)
    h0_strategy: FixNoiseStrategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
    trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory, setting.device)

    # Load the pre-trained model
    model_path = "saved_models/pretrained.pt"  # Update this path if needed
    checkpoint = torch.load(model_path, map_location=setting.device, weights_only=True)
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    h0_strategy.h0 = checkpoint["h0"]
    print("Loaded pre-trained model from", model_path)

    # # Print the model's state_dict
    # print("Model's state_dict:")
    # for param_tensor in trainer.model.state_dict():
    #     print(param_tensor, "\t", trainer.model.state_dict()[param_tensor].size())

    # Initialize the evaluation
    evaluation_test = Evaluation(dataset_test, dataloader_test, poi_loader.user_count(), h0_strategy, trainer, setting)

    # Evaluate the model
    print("=================== Test Set Evaluation ===================")
    import psutil
    print(f"Memory usage: {psutil.virtual_memory().percent}%") #! WARNING: Can take a while if dataset is huge
    evaluation_test.evaluate()


if __name__ == "__main__":
    main()

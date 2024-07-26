from models.GPT_Model import MidiGPT
from data.GPT_Dataset import MidiGPT_Dataset
from torch.utils.data import DataLoader
import torch

from datetime import datetime
from tqdm import tqdm
import logging
from time import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Hyperparameters:
batch_size = 64
block_size = 256
epochs = 1000
learning_rate = 1e-4
n_embd_pitch = 64
n_embd_velocity = 9
n_head = 5
n_layer = 5
dropout = 0.1

train_data_json_dir = './data/train_data.json'
test_data_json_dir = './data/test_data.json'

# Set up logging
logging_dir = "./logs/" + str(datetime.now().date()) + "/"
os.makedirs(logging_dir, exist_ok=True)
logger = logging.getLogger(__name__)
logging.basicConfig(filename=logging_dir + f"train_log_{datetime.now().time()}.txt".replace(":", "."), level=logging.INFO)

# This prints it to the screen as well
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)

logging.info(f"Generation Task")
logging.info(f"Using device: {device}")

model = MidiGPT(
            pitch_size=130,
            velocity_size=130,
            n_embed_pitch=n_embd_pitch,
            n_embed_velocity=n_embd_velocity,
            n_layer=n_layer,
            n_head=n_head,
            block_size=block_size,
            dropout=dropout
            ).to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

train_dataset = MidiGPT_Dataset(json_dir=train_data_json_dir, block_size=block_size, debug=False)
train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MidiGPT_Dataset(json_dir=test_data_json_dir, block_size=block_size)
test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_loss = float('inf')

for e in range(epochs):

    train_loss_sum = 0
    test_loss_sum = 0

    start_time = time()

    model.train()
    for batch in tqdm(train_dl, desc='Train', ascii=True):

        src, tgt = batch['src'].to(device), batch['tgt'].to(device)

        logits, loss = model(src, tgt)
        train_loss_sum += loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    for batch in tqdm(test_dl, desc='Test', ascii=True):

        src, tgt = batch['src'].to(device), batch['tgt'].to(device)

        # evaluate the loss
        logits, loss = model(src, tgt)
        test_loss_sum += loss.item()

    epoch_time = time() - start_time

    logging.info(f"Epoch {e}:\n\tAVG Train Loss:\t{train_loss_sum/len(train_dl):.4f}\n\tAVG Test Loss:\t{test_loss_sum/len(test_dl):.4f}")
    logging.info(f"Time: {epoch_time} seconds, ETA: {(epochs-e)*epoch_time} seconds")

    if test_loss_sum/len(test_dl) < best_loss:
        best_loss = test_loss_sum/len(test_dl)
        logging.info("New best model, saving...")
        torch.save(model.state_dict(), logging_dir + "best_gpt.pt")
    torch.save(model.state_dict(), logging_dir + "last_gpt.pt")

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# #open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
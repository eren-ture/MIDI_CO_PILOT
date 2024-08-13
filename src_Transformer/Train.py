from argparse import ArgumentParser
from Dataset import MidiDataset
from torch.utils.data import DataLoader
from TransformerModel import MidiTransformer
import torch
from tqdm import tqdm
import logging
import os
from datetime import datetime
from time import time
from config import DEVICE
import config

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_dl, optim, opts):

    # Objects for accumulating losses
    losses = 0
    time_l = 0
    duration_l = 0
    pitch_l = 0
    velocity_l = 0

    # Put model into training mode
    model.train()
    for batch in tqdm(train_dl, ascii=True):

        src = batch['cxt_input'].float().to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].to(DEVICE)
        tgt_mask = batch['tgt_mask'].to(DEVICE)
        src_padding_mask = batch['cxt_padding_mask'].to(DEVICE)
        tgt_padding_mask = batch['tgt_padding_mask'].to(DEVICE)

        print(f"Source:\n{src[0]}")
        print(f"Target:\n{tgt[0]}")

        out, loss, t_loss, d_loss, p_loss, v_loss = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        print(f"Output:\n{out[0]}")

        # show_src, show_tgt, show_out = src[0].cpu().detach().numpy(), tgt[0].cpu().detach().numpy(), logits[0].cpu().detach().numpy()
        # file_nm = i

        # np.save(f"train_out\\{file_nm}_src.npy", show_src)
        # np.save(f"train_out\\{file_nm}_tgt.npy", show_tgt)
        # np.save(f"train_out\\{file_nm}_out.npy", show_out)

        # i += 1

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()
        time_l += t_loss.item()
        duration_l += d_loss.item()
        pitch_l += p_loss.item()
        velocity_l += v_loss.item()

        if opts.dry_run:
            break

    return (
        losses / len(train_dl),
        time_l / len(train_dl),
        duration_l / len(train_dl),
        pitch_l / len(train_dl),
        velocity_l / len(train_dl),
        )

# Check the model accuracy on the validation dataset
@torch.no_grad()
def validate(model, valid_dl):

    # Object for accumulating losses
    losses = 0
    time_l = 0
    duration_l = 0
    pitch_l = 0
    velocity_l = 0

    # Turn off gradients
    model.eval()
    for batch in tqdm(valid_dl):

        src = batch['cxt_input'].float().to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].float().to(DEVICE)
        src_padding_mask = batch['cxt_padding_mask'].float().to(DEVICE)
        tgt_padding_mask = batch['tgt_padding_mask'].float().to(DEVICE)

        # Pass into model, get probability over the vocab out
        _, loss, t_loss, d_loss, p_loss, v_loss = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        losses += loss.item()
        time_l += t_loss.item()
        duration_l += d_loss.item()
        pitch_l += p_loss.item()
        velocity_l += v_loss.item()

    # Return the average loss
    return (
        losses / len(valid_dl),
        time_l / len(valid_dl),
        duration_l / len(valid_dl),
        pitch_l / len(valid_dl),
        velocity_l / len(valid_dl),
        )

# Train the model
def main(opts):

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + f"train_log_{datetime.now().time()}.txt".replace(":", "."), level=logging.INFO)

    # This prints it to the screen as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    logging.info(f"Generation Task")
    logging.info(f"Using device: {DEVICE}")

    # Get training data, tokenizer and vocab
    # objects as well as any special symbols we added to our dataset

    train_dataset = MidiDataset(json_dir=opts.train_data_json_dir, max_seq_len=config.max_seq_len, debug=False)
    train_dl = DataLoader(train_dataset, batch_size=opts.batch, shuffle=True)

    test_dataset = MidiDataset(json_dir=opts.test_data_json_dir, max_seq_len=config.max_seq_len)
    test_dl = DataLoader(test_dataset, batch_size=opts.batch, shuffle=True)

    logging.info("Loaded data")

    # src_vocab_size = len(src_vocab)
    # tgt_vocab_size = len(tgt_vocab)

    # logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    # logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = MidiTransformer(**config.hyperparameters).to(DEVICE)

    logging.info("Model created... starting training!")

    if opts.model_path:
        logging.info(f"Model loaded from: {opts.model_path}")
        model.load_state_dict(torch.load(opts.model_path))

    logging.info(f"Number of parameters: {count_parameters(model)/1e6}M")

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=opts.lr, max_lr=0.1, step_size_up=252)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss, train_time_l, train_drtn_l, train_ptch_l, train_velo_l = train(model, train_dl, optim, opts)
        epoch_time = time() - start_time
        test_loss, test_time_l, test_drtn_l, test_ptch_l, test_velo_l = validate(model, test_dl)

        # Once training is done, we want to save out the model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best_transformer.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last_transformer.pt")

        # logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {test_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds")
        logger.info(f"""Epoch: {epoch}
    Total Train Loss: {train_loss:.3f}
        Time:   {train_time_l:.3f}
        Duration:   {train_drtn_l:.3f}
        Pitch:  {train_ptch_l:.3f}
        Velocity:   {train_velo_l:.3f}
    Total Test Loss: {test_loss:.3f}
        Time:   {test_time_l:.3f}
        Duration:   {test_drtn_l:.3f}
        Pitch:  {test_ptch_l:.3f}
        Velocity:   {test_velo_l:.3f}
    Epoch time = {epoch_time:.1f} seconds
    ETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds""")

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Midi Co-Pilot training",
    )

    # Continue training
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")
    
    # Dataset settings
    parser.add_argument("--train_data_json_dir", type=str, default='../data/train_data.json',
                        help="train_data.json directory")
    parser.add_argument("--test_data_json_dir", type=str, default='../data/test_data.json',
                        help="test_data.json directory")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=200,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./logs/" + str(datetime.now().date()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    main(args)
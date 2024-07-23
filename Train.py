from argparse import ArgumentParser
from data.Dataset import MidiDataset
from torch.utils.data import DataLoader, TensorDataset
from symusic import Score
from models.TransformerModel import MidiTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os
from datetime import datetime
from time import time
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def midi_loss_fn(output, target):
#     '''
#     MSE Loss
#     '''
#     out_mask = output[...,0].isnan()
#     tgt_mask = target[...,0] == -2
#     mask = ~(out_mask | tgt_mask)

#     out_times,      tgt_times       = output[...,0][mask], target[...,0][mask]
#     out_durations,  tgt_durations   = output[...,1][mask], target[...,1][mask]
#     out_pitches,    tgt_pitches     = output[...,2][mask], target[...,2][mask]
#     out_velocities, tgt_velocities  = output[...,3][mask], target[...,3][mask]

#     loss = F.mse_loss(out_times, tgt_times) + F.mse_loss(out_durations, tgt_durations) + F.mse_loss(out_pitches, tgt_pitches) + F.mse_loss(out_velocities, tgt_velocities)

#     print(f"\tLoss: {loss.item()}")
#     return loss

def midi_loss_fn(output, target):
    '''
    MSE + Cross Entropy Loss
    '''
    out_mask = output.isnan()
    tgt_mask = target == -2
    mask = ~(out_mask | tgt_mask)

    valid_indices = mask.all(dim=-1)

    out_times,      tgt_times       = output[...,0][valid_indices], target[...,0][valid_indices]
    out_durations,  tgt_durations   = output[...,1][valid_indices], target[...,1][valid_indices]

    out_pitches,    tgt_pitches     = output[...,2:130][valid_indices], target[...,2:130][valid_indices]
    out_velocities, tgt_velocities  = output[...,130:258][valid_indices], target[...,130:258][valid_indices]

    time_loss =     F.mse_loss(out_times, tgt_times)
    duration_loss = F.mse_loss(out_durations, tgt_durations)
    pitch_loss =    F.cross_entropy(out_pitches.reshape(-1, 128), torch.argmax(tgt_pitches.reshape(-1, 128), dim=1))
    velocity_loss = F.cross_entropy(out_velocities.reshape(-1, 128), torch.argmax(tgt_velocities.reshape(-1, 128), dim=1))

    # np.save("train_out\\pitches_in_loss.npy", tgt_pitches.reshape(-1, 128).cpu().detach().numpy())
    # np.save("train_out\\velocities_in_loss.npy", tgt_velocities.reshape(-1, 128).cpu().detach().numpy())

    loss = (.005 * time_loss) + duration_loss + pitch_loss + velocity_loss
    return (
        loss + 1e-8,        # Total Loss
        .005 * time_loss,   # Time Loss
        duration_loss,      # Duration Loss
        pitch_loss,         # Pitch Loss
        velocity_loss       # Velocity Loss
    )

def train(model, train_dl, loss_fn, optim, opts):

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
        src_mask = batch['cxt_mask'].to(DEVICE)

        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].to(DEVICE)

        logits = model(src, tgt, src_mask, tgt_mask)

        # show_src, show_tgt, show_out = src[0].cpu().detach().numpy(), tgt[0].cpu().detach().numpy(), logits[0].cpu().detach().numpy()
        # file_nm = i

        # np.save(f"train_out\\{file_nm}_src.npy", show_src)
        # np.save(f"train_out\\{file_nm}_tgt.npy", show_tgt)
        # np.save(f"train_out\\{file_nm}_out.npy", show_out)

        # i += 1

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        loss, time, drtn, ptch, velo = loss_fn(logits, tgt)
        loss.backward()

        # Step weights
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()
        time_l += time.item()
        duration_l += drtn.item()
        pitch_l += ptch.item()
        velocity_l += velo.item()

        if opts.dry_run:
            break

    # Return the average loss
    return (
        losses / len(train_dl),
        time_l / len(train_dl),
        duration_l / len(train_dl),
        pitch_l / len(train_dl),
        velocity_l / len(train_dl),
        )

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn):

    # Object for accumulating losses
    losses = 0
    time_l = 0
    duration_l = 0
    pitch_l = 0
    velocity_l = 0

    # Turn off gradients a moment
    model.eval()

    for batch in tqdm(valid_dl):

        src = batch['cxt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].to(DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt, src_mask, tgt_mask)
        loss, time, drtn, ptch, velo = loss_fn(logits, tgt)

        losses += loss.item()
        time_l += time.item()
        duration_l += drtn.item()
        pitch_l += ptch.item()
        velocity_l += velo.item()

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

    train_dataset = MidiDataset(json_dir=opts.train_data_json_dir, max_seq_len=opts.max_seq_len, debug=False)
    train_dl = DataLoader(train_dataset, batch_size=opts.batch, shuffle=True)

    test_dataset = MidiDataset(json_dir=opts.test_data_json_dir, max_seq_len=opts.max_seq_len)
    test_dl = DataLoader(test_dataset, batch_size=opts.batch, shuffle=True)

    logging.info("Loaded data")

    # src_vocab_size = len(src_vocab)
    # tgt_vocab_size = len(tgt_vocab)

    # logging.info(f"{opts.src} vocab size: {src_vocab_size}")
    # logging.info(f"{opts.tgt} vocab size: {tgt_vocab_size}")

    # Create model
    model = MidiTransformer(
        embed_size=(2+128+128),
        num_encoder_layers=opts.enc_layers,
        num_decoder_layers=opts.dec_layers,
        num_heads=opts.attn_heads,
        dim_feedforward=opts.dim_feedforward,
        dropout=opts.dropout
        ).to(DEVICE)

    logging.info("Model created... starting training!")

    if opts.model_path:
        logging.info(f"Model loaded from: {opts.model_path}")
        model.load_state_dict(torch.load(opts.model_path))

    logging.info(f"Number of parameters: {count_parameters(model)}")

    # Set up our learning tools
    loss_fn = midi_loss_fn

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss, train_time_l, train_drtn_l, train_ptch_l, train_velo_l = train(model, train_dl, loss_fn, optim, opts)
        epoch_time = time() - start_time
        test_loss, test_time_l, test_drtn_l, test_ptch_l, test_velo_l = validate(model, test_dl, loss_fn)

        # Once training is done, we want to save out the model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")

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
        prog="Machine Translator training and inference",
    )

    # Continue training
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")
    
    # Dataset settings
    parser.add_argument("--train_data_json_dir", type=str, default='./data/train_data.json',
                        help="train_data.json directory")
    parser.add_argument("--test_data_json_dir", type=str, default='./data/test_data.json',
                        help="test_data.json directory")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Size of the target/context sequences")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=200,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="gpu",
                        help="Batch size")
    
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=6,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--embed_size", type=int, default=512,
                        help="Size of the language embedding")
    parser.add_argument("--dim_feedforward", type=int, default=512,
                        help="Feedforward dimensionality")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Transformer dropout")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./logs/" + str(datetime.now().date()) + "/",
                        help="Where the output of this program should be placed")

    # Just for continuous integration
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    DEVICE = torch.device("cuda" if args.backend == "gpu" and torch.cuda.is_available() else "cpu")

    main(args)
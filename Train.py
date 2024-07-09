from argparse import ArgumentParser
from data.Dataset import MidiDataset
from torch.utils.data import DataLoader, TensorDataset
from symusic import Score
from models.TransformerModel import MidiTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
import logging
import os
from datetime import datetime
from time import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mse(out, tgt):
    return torch.mean((out-tgt)**2)

def midi_loss_fn(output, target):
    out_mask = output[...,0].isnan()
    tgt_mask = target[...,0] == -2
    mask = ~(out_mask | tgt_mask)

    out_times,      tgt_times       = output[...,0][mask], target[...,0][mask]
    out_durations,  tgt_durations   = output[...,1][mask], target[...,1][mask]
    out_pitches,    tgt_pitches     = output[...,2][mask], target[...,2][mask]
    out_velocities, tgt_velocities  = output[...,3][mask], target[...,3][mask]

    loss = mse(out_times, tgt_times) + mse(out_durations, tgt_durations) + mse(out_pitches, tgt_pitches) + mse(out_velocities, tgt_velocities)

    print(f"\tLoss: {loss.item()}")
    return loss

# def midi_loss_fn(output, target, tgt_mask):
#     diff_sum = 0
#     non_nan_values = 0

#     batch_size, seq_len, features = output.shape
#     mask = ~tgt_mask  # Invert the mask to get non-padding positions

#     for b in range(batch_size):
#         for i in range(seq_len):
#             if mask[b, i]:
#                 for f in range(features):
#                     if not torch.isnan(output[b, i, f]) and not torch.isnan(target[b, i, f]):
#                         non_nan_values += 1
#                         diff_sum += (output[b, i, f] - target[b, i, f])**2

#     if non_nan_values == 0:
#         return torch.tensor(0.0, requires_grad=True)

#     epsilon = 1e-8
#     loss = diff_sum / (non_nan_values + epsilon)
#     print(f"\tLoss: {loss.item()}")
#     return loss


def train(model, train_dl, loss_fn, optim, opts):

    # Object for accumulating losses
    losses = 0

    # Put model into training mode
    model.train()
    for batch in tqdm(train_dl, ascii=True):

        src = batch['cxt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].to(DEVICE)

        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].to(DEVICE)

        # print(f"Source nan values: {src.isnan().sum()}")
        # print(f"Target nan values: {tgt.isnan().sum()}")

        # print("Checking for NaNs in gradients before step...")
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         if torch.isnan(param.grad).any():
        #             print(f"NaNs detected in gradients of {name}")

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt, src_mask, tgt_mask)

        print("Checking for NaNs in intermediate outputs...")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaNs detected in {name}")

        debug_logit, debug_target = logits[0], tgt[0]
        print("\tTarget:")
        print(debug_target)
        print("\tOutput:")
        print(debug_logit)

        # padding = torch.tensor([-2, -2, -2, -2])
        # n_null, n_padding = torch.isnan(debug_logit).sum(), (debug_target == padding).sum()

        # print(f"\tNull seq: {n_null}")
        # print(f"\tPad seq: {n_padding}\n")

        # print(f"Logits Shape:\t\t{logits.shape}")
        # print(f"Target Shape:\t\t{tgt.shape}")
        # print(f"Target Mask Shape:\t{tgt_mask.shape}")

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        # tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        # loss = loss_fn(torch.flatten(logits), torch.flatten(tgt))
        loss = loss_fn(logits, tgt)
        # loss.backward()

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, retain_graph=True)

        # Apply gradients, ignoring NaN values
        n_nan = 0
        for grad, param in zip(grads, model.parameters()):
            if torch.isnan(grad).any():
                n_nan += 1
                continue
            param.grad = grad

        print(f"{n_nan}/{count_parameters(model)} parameters are NaN")

        print("Checking for NaNs in gradients...")
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"NaNs detected in gradients of {name}")

        # Step weights
        optim.step()

        # Accumulate a running loss for reporting
        losses += loss.item()

        if opts.dry_run:
            break

    # Return the average loss
    return losses / len(list(train_dl))

# Check the model accuracy on the validation dataset
def validate(model, valid_dl, loss_fn):

    # Object for accumulating losses
    losses = 0

    # Turn off gradients a moment
    model.eval()

    for batch in tqdm(valid_dl):

        src = batch['cxt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        # tgt_input = tgt[:-1, :]

        # Create masks
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt, src_mask, tgt_mask)#,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        # tgt_out = tgt[1:, :]
        # loss = loss_fn(logits.flatten(), tgt.flatten())
        loss = loss_fn(logits, tgt)
        losses += loss.item()

    # Return the average loss
    return losses / len(list(valid_dl))

# Train the model
def main(opts):

    # Set up logging
    os.makedirs(opts.logging_dir, exist_ok=True)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=opts.logging_dir + f"log_{datetime.now().time()}.txt", level=logging.INFO)

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
        embed_size=4,
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
    # loss_fn = torch.nn.MSELoss()
    loss_fn = midi_loss_fn

    # These special values are from the "Attention is all you need" paper
    optim = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(0.9, 0.98), eps=1e-9)

    best_val_loss = 1e6
    
    for idx, epoch in enumerate(range(1, opts.epochs+1)):

        start_time = time()
        train_loss = train(model, train_dl, loss_fn, optim, opts)
        epoch_time = time() - start_time
        test_loss  = validate(model, test_dl, loss_fn)

        # Once training is done, we want to save out the model
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            logging.info("New best model, saving...")
            torch.save(model.state_dict(), opts.logging_dir + "best.pt")

        torch.save(model.state_dict(), opts.logging_dir + "last.pt")

        logger.info(f"Epoch: {epoch}\n\tTrain loss: {train_loss:.3f}\n\tVal loss: {test_loss:.3f}\n\tEpoch time = {epoch_time:.1f} seconds\n\tETA = {epoch_time*(opts.epochs-idx-1):.1f} seconds")

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Machine Translator training and inference",
    )

    # Inference mode
    parser.add_argument("--inference", action="store_true",
                        help="Set true to run inference")
    parser.add_argument("--model_path", type=str,
                        help="Path to the model to run inference on")

    # Translation settings
    parser.add_argument("--src", type=str, default="de",
                        help="Source language (translating FROM this language)")
    parser.add_argument("--tgt", type=str, default="en",
                        help="Target language (translating TO this language)")
    
    # Dataset settings
    parser.add_argument("--train_data_json_dir", type=str, default='./data/train_data.json',
                        help="train_data.json directory")
    parser.add_argument("--test_data_json_dir", type=str, default='./data/test_data.json',
                        help="test_data.json directory")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Size of the target/context sequences")

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="cpu",
                        help="Batch size")
  
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=6,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=6,
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

    if args.inference:
        inference(args)
    else:
        main(args)
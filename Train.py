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

def train(model, train_dl, loss_fn, optim, opts):

    # Object for accumulating losses
    losses = 0

    # Put model into training mode
    model.train()
    for batch in tqdm(train_dl, ascii=True):

        src = batch['cxt_input'].float().to(DEVICE)
        src_mask = batch['cxt_mask'].t().to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].t().to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        # tgt_input = tgt[:-1, :]

        # Create masks
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, [-2 , -2, -2, -2], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt, src_mask, tgt_mask)
        print(logits)
        print(f"Logits shape: {logits.shape}")
        print(f"target shape: {tgt.shape}")

        # Reset gradients before we try to compute the gradients over the loss
        optim.zero_grad()

        # Get original shape back
        # tgt_out = tgt[1:, :]

        # Compute loss and gradient over that loss
        flat_logits, flat_target = torch.flatten(logits), torch.flatten(tgt)
        print(f"Flat Logits shape: {flat_logits.shape}")
        print(f"Flat Target shape: {flat_target.shape}")
        loss = loss_fn(flat_logits, flat_target)
        loss.backward()

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
        src_mask = batch['cxt_mask'].t().to(DEVICE)
        tgt = batch['tgt_input'].float().to(DEVICE)
        tgt_mask = batch['tgt_mask'].t().to(DEVICE)

        # We need to reshape the input slightly to fit into the transformer
        # tgt_input = tgt[:-1, :]

        # Create masks
        # src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, special_symbols["<pad>"], DEVICE)

        # Pass into model, get probability over the vocab out
        logits = model(src, tgt, src_mask, tgt_mask)#,src_padding_mask, tgt_padding_mask, src_padding_mask)

        # Get original shape back, compute loss, accumulate that loss
        # tgt_out = tgt[1:, :]
        loss = loss_fn(logits.flatten(), tgt.flatten())
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

    train_dataset = MidiDataset(json_dir=opts.train_data_json_dir, max_seq_len=512)
    train_dl = DataLoader(train_dataset, batch_size=opts.batch, shuffle=True)

    test_dataset = MidiDataset(json_dir=opts.test_data_json_dir, max_seq_len=512)
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

    # Set up our learning tools
    loss_fn = torch.nn.MSELoss()

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

    # Training settings 
    parser.add_argument("-e", "--epochs", type=int, default=30,
                        help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Default learning rate")
    parser.add_argument("--batch", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--backend", type=str, default="cpu",
                        help="Batch size")
  
    # Transformer settings
    parser.add_argument("--attn_heads", type=int, default=2,
                        help="Number of attention heads")
    parser.add_argument("--enc_layers", type=int, default=5,
                        help="Number of encoder layers")
    parser.add_argument("--dec_layers", type=int, default=5,
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
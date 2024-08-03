from argparse import ArgumentParser
from TransformerModel import MidiTransformer
import torch
import symusic
from config import DEVICE
import config
from datetime import datetime
from utils import quantize_to_32nds, track_to_list
import numpy as np
import os

def create_time_weighted_attention_mask(times, decay=0.2, device=DEVICE):
    """
    Create a time-weighted attention mask for a single sequence where future tokens are masked out
    and the remaining tokens are weighted based on their relative time to the last attended token,
    with more recent tokens receiving higher weights.

    Parameters:
    - times (torch.Tensor): The times for each token in the sequence, shape [seq_length].
    - decay (float): The decay rate for the weights based on relative time.
    - device (torch.device): The device to run the calculations on.

    Returns:
    - torch.Tensor: The time-weighted attention mask, shape [seq_length, seq_length].
    """
    if device is None:
        device = torch.device('cpu')

    times = times.to(device)

    seq_length = times.size(0)
    causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
    
    # Expand times to [seq_length, seq_length] for broadcasting
    times_expanded = times.unsqueeze(0).expand(seq_length, -1)
    
    # Calculate the relative times for each pair of tokens
    relative_times = torch.abs(times_expanded - times_expanded.t())  # [seq_length, seq_length]

    # Apply decay function to the relative times
    time_weights = torch.exp(-decay * relative_times)  # [seq_length, seq_length]
    
    # Apply the causal mask
    weighted_mask = time_weights * causal_mask
    weighted_mask = weighted_mask.masked_fill(causal_mask == 0, float('-inf'))
    
    return weighted_mask


def midi_to_src(midi):

    for track in midi.tracks:

        source = []

        # Quantize notes
        for note in track.notes:
            note.time = quantize_to_32nds(note.time)
            note.duration = quantize_to_32nds(note.duration)

        # Append track to source
        source.expand(track_to_list(track))

    # Pad or truncate source
    seq_len = len(source)
    if seq_len >= config.max_seq_len:
        source = torch.as_tensor(source[:config.max_seq_len])
        pad_mask = torch.zeros(config.max_seq_len)
    else:
        p_note = [0, 0, config.symbols['pad'], config.symbols['pad']]
        padding = [p_note] * (config.max_seq_len - seq_len)
        mask = torch.concatenate((torch.zeros(seq_len), torch.ones(config.max_seq_len - seq_len)))
        source = torch.as_tensor(source + padding)
        pad_mask = mask.masked_fill(mask == 1, float('-inf'))

    return source, pad_mask


@torch.no_grad()
def infer(model, midi, max_len = config.max_seq_len):

    model.eval()

    src = midi_to_src(midi).to(DEVICE)
    T, C = src.shape

    src_mask = torch.zeros(T, T).type(float).to(DEVICE)

    memory = model.encode(src, src_mask).to(DEVICE)

    ys = torch.as_tensor([[0, 0, config.symbols['bot'], config.symbols['bot']]]).type(torch.float).to(DEVICE)

    for _ in range(max_len-1):

        tgt_mask = create_time_weighted_attention_mask(ys[:, 0], decay=config.attn_mask_time_weight_decay)
        outs = model.decode(ys, memory, tgt_mask)       # T, embed_size
        out = outs[:-1]                                 # embed_size

        out_time = model.time_ff(out)                   # 1
        out_time = model.relu(out_time)                 # 1
        out_time = out_time.item()

        out_duration = model.duration_ff(out)           # 1
        out_duration = model.relu(out_duration)         # 1
        out_duration = out_duration.item()

        out_pitch = model.pitch_ff(out)                 # 131
        out_pitch = torch.argmax(out_pitch)

        out_velocity = model.velocity_ff(out)           # 131
        out_velocity = torch.argmax(out_velocity)

        next_note = torch.as_tensor([out_time, out_duration, out_pitch, out_velocity]).type(torch.float)

        ys = torch.cat([ys, next_note], dim=0)
        if (out_pitch >= config.symbol['eot']) or (out_velocity >= config.symbol['eot']):
            break

    return ys
    

def main(opts):

    os.makedirs(opts.output_path, exist_ok=True)

    model = MidiTransformer(**config.hyperparameters).to(DEVICE)
    model.load_state_dict(torch.load(opts.model_path))

    midi = symusic.Score(opts.midi_path)

    output = infer(model, midi, max_len=config.max_seq_len)

    notes_list = output.cpu().detach().numpy()
    notes_list.dump(os.path.join(opts.output_path, 'test.npy'))

    return 0

if __name__ == "__main__":

    parser = ArgumentParser(
        prog="Midi Co-Pilot inference",
    )

    # Model Path
    parser.add_argument("model_path", type=str,
                        help="Path to the model to run inference on")
    
    # I/O Path
    parser.add_argument("midi_path", type=str,
                        help="Path to the input midi")
    parser.add_argument("--output_path", type=str, default="./inference_output/",
                        help="Path of the output folder")

    # Logging settings
    parser.add_argument("--logging_dir", type=str, default="./logs/" + str(datetime.now().date()) + "/",
                        help="Where the output of this program should be placed")

    args = parser.parse_args()

    main(args)
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import track_to_list, get_clipped_tracks
from symusic import Score
import numpy as np
import random
import torch
import json
import os
import logging
from datetime import datetime
from config import DEVICE
import config

class MidiDataset(Dataset):
    def __init__(self, json_dir, max_seq_len, debug=False):
        with open(json_dir, 'r') as data_json:
            self.data = json.load(data_json)
        self.max_seq_len = max_seq_len

        self.symbols = config.symbols

        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        midi_dir = "." + data_point['Midi_Directory']
        artist = data_point['Artist']
        song_name = data_point['Song']

        midi = Score(midi_dir, ttype='quarter')
        n_bars = random.choice([4, 8, 16]) # Randomly choose amount of bars.
        if self.debug:
            self.logger.info(f"Processing: {midi_dir}, with {n_bars} bars splits.")

        tracks = get_clipped_tracks(
            midi,
            n_bars=n_bars, 
            zero_index=True,
            debug=False
            )
        if not tracks:
            self.logger.info(f"\tWarning: {midi_dir}: Bar length too short.\n\t{midi.time_signatures[0]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        n_tracks = len(tracks)
        
        # Randomly select a target track
        target_ix = random.choice(range(n_tracks))
        tgt_parts = tracks[target_ix]
        context = [tracks[i] for i in range(n_tracks) if i != target_ix]

        pairs = []

        # Go through all target parts, if they have corresponding context, add them as part data.
        for i, tgt_part in enumerate(tgt_parts):
            if len(tgt_part) <= 0:  # Skip empty parts.
                continue
            
            non_empty_parts = []
            for cxt_parts in context:
                if len(cxt_parts[i]) <= 0:  # Skip empty contexts.
                    continue
                non_empty_parts.append(cxt_parts[i])

            if non_empty_parts:
                pairs.append((tgt_part, non_empty_parts))

        if pairs:

            # Choose a random context-target pair.
            tgt_part, cxt_parts = random.choice(pairs)

            tgt, tgt_padding_mask = self.pad_or_truncate(self.bot_eot_target(tgt_part), tgt=True)
            cxt, cxt_padding_mask = self.pad_or_truncate(self.concat_sequences(cxt_parts))

            self.pre_process(tgt)
            self.pre_process(cxt)

            tgt_attn_mask = self.create_time_weighted_attention_mask(tgt[...,0], decay=config.attn_mask_time_weight_decay, device=DEVICE)
            cxt_attn_mask = torch.zeros((self.max_seq_len, self.max_seq_len)).type(torch.bool)

            return {
                'tgt_input' : tgt,
                'cxt_input' : cxt,
                'tgt_mask' : tgt_attn_mask,
                'cxt_mask' : cxt_attn_mask,
                'tgt_padding_mask' : tgt_padding_mask,
                'cxt_padding_mask' : cxt_padding_mask,
                'artist': artist,
                'song_name': song_name
            }
        
        else:
            if self.debug:
                self.logger.info(f"\tWarning: No non-empty parts found.")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def concat_sequences(self, sequences):
        res = []
        for seq in sequences:
            res.extend(seq)
        res.sort(key=lambda x: x[0])
        return res

    def pad_or_truncate(self, sequence, tgt=False):
        seq_len = len(sequence)
        if seq_len >= self.max_seq_len:
            if tgt:
                e_note = [0, 0, self.symbols['eot'], self.symbols['eot']]   # End note
                return (torch.as_tensor(sequence[:self.max_seq_len-1] + [e_note]), torch.zeros(self.max_seq_len))
            else:
                return (torch.as_tensor(sequence[:self.max_seq_len]), torch.zeros(self.max_seq_len))
        else:
            p_note = [0, 0, self.symbols['pad'], self.symbols['pad']]
            padding = [p_note] * (self.max_seq_len - seq_len)
            mask = torch.concatenate((torch.zeros(seq_len), torch.ones(self.max_seq_len - seq_len)))
            return (torch.as_tensor(sequence + padding), mask.masked_fill(mask == 1, float('-inf')))

    def bot_eot_target(self, sequence):
        b_note = [0, 0, self.symbols['bot'], self.symbols['bot']]   # Begining note
        e_note = [0, 0, self.symbols['eot'], self.symbols['eot']]   # End note
        return [b_note] + sequence + [e_note]
    
    def generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def pre_process(self, x):
        '''
        Clamps duration values down to 4.
        '''
        x[...,1] = x[...,1].clamp(max=4.)
    
    def create_time_weighted_attention_mask(self, times, decay=0.2, device=DEVICE):
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
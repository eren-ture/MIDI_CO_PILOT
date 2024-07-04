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
class MidiDataset(Dataset):
    def __init__(self, json_dir, max_seq_len, debug=False):
        with open(json_dir, 'r') as data_json:
            self.data = json.load(data_json)
        self.max_seq_len = max_seq_len

        self.debug = debug
        if self.debug:
            self.logger = logging.getLogger(__name__)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        midi_dir = data_point['Midi_Directory']
        artist = data_point['Artist']
        song_name = data_point['Song']

        midi = Score(midi_dir)
        n_bars = random.choice([4, 8, 16]) # Randomly choose amount of bars.
        if self.debug:
            self.logger.info(f"Processing: {midi_dir}, with {n_bars} bars splits.")

        tracks = list(get_clipped_tracks(
            midi,
            n_bars=n_bars, 
            zero_index=True,
            ).items())
        n_tracks = len(tracks)
        
        # Randomly select a target track
        target_ix = random.choice(range(n_tracks))
        tgt_name, tgt_parts = tracks[target_ix]
        context = [tracks[i] for i in range(n_tracks) if i != target_ix]

        pairs = []

        # Go through all target parts, if they have corresponding context, add them as part data.
        for i, tgt_part in enumerate(tgt_parts):
            if len(tgt_part) <= 0:  # Skip empty parts.
                continue
            
            non_empty_parts = []
            for cxt_name, cxt_parts in context:
                if len(cxt_parts[i]) <= 0:  # Skip empty contexts.
                    continue
                non_empty_parts.append(cxt_parts[i])

            if non_empty_parts:
                pairs.append((tgt_part, non_empty_parts))

        if pairs:

            # Choose a random context-target pair.
            tgt_part, cxt_parts = random.choice(pairs)

            # Pad or truncate
            if self.debug:
                self.logger.info("\tTarget:")
            tgt_input, tgt_mask = self.pad_or_truncate(tgt_part)

            if self.debug:
                self.logger.info("\tContext:")
            cxt_input, cxt_mask = self.pad_or_truncate(self.concat_sequences(cxt_parts))

            return {
                'tgt_input':    torch.as_tensor(tgt_input),
                'tgt_mask':     torch.as_tensor(tgt_mask),
                'cxt_input':    torch.as_tensor(cxt_input),
                'cxt_mask':     torch.as_tensor(cxt_mask),
                'artist': artist,
                'song_name': song_name
            }
        
        else:

            if self.debug:
                self.logger.info(f"No non-empty parts found.")

            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def concat_sequences(self, sequences):
        res = []
        for seq in sequences:
            res.extend(seq)
        res.sort(key=lambda x: x[0])
        return res

    def pad_or_truncate(self, sequence):
        '''
        Returns (Padded/Truncated sequence, padding mask)
        '''
        sequence_length = len(sequence)
        if sequence_length > self.max_seq_len:
            if self.debug:
                self.logger.info(f"\t\tTruncated, {sequence_length}")

            return (sequence[:self.max_seq_len], np.zeros(self.max_seq_len).astype(bool))
        else:
            if self.debug:
                self.logger.info("\t\tPadded")

            padding = [[-2, -2, -2, -2]] * (self.max_seq_len - sequence_length)
            mask = np.concatenate((np.zeros(sequence_length), np.ones(self.max_seq_len - sequence_length))).astype(bool)
            return (sequence + padding, mask)
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import track_to_list, get_clipped_tracks
from symusic import Score
import numpy as np
import random
import torch
import json
import os
import logging

class MidiGPT_Dataset(Dataset):
    def __init__(self, json_dir, block_size, debug=False):
        with open(json_dir, 'r') as data_json:
            self.data = json.load(data_json)
        self.block_size = block_size

        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_point = self.data[idx]
        midi_dir = data_point['Midi_Directory']
        artist = data_point['Artist']
        song_name = data_point['Song']

        midi = Score(midi_dir, ttype='quarter')
        n_bars = random.choice([8, 16]) # Randomly choose amount of bars.
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

            # Pad and choose a random point for the block start.
            seq = self.pad_sequence(self.concat_sequences(cxt_parts, tgt_part))
            ix = random.randint(0, len(seq)-self.block_size-1) if len(seq)-self.block_size-1 > 0 else 0
            src = torch.as_tensor(seq[ix:ix+self.block_size])
            tgt = torch.as_tensor(seq[ix+1:ix+self.block_size+1])

            return {
                'src' : src,
                'tgt' : tgt,
                'artist' : artist,
                'song_name' : song_name
            }
        
        else:
            if self.debug:
                self.logger.info(f"\tWarning: No non-empty parts found.")
            return self.__getitem__(random.randint(0, len(self) - 1))
    
    def concat_sequences(self, cxts, tgt):
        eom = [0, 0, 128, 128]    # End of Midi token
        res = []
        random.shuffle(cxts)
        for seq in cxts:
            res.extend(seq)
            res.append(eom)
        res.extend(tgt)
        res.append(eom)
        return res

    def pad_sequence(self, sequence):
        pad = [0, 0, 129, 129]    # Padding token
        seq_len = len(sequence)
        if seq_len <= self.block_size:
            if self.debug:
                self.logger.info(f"\tWarning: Padded sequence.")
            padding = [pad] * (self.block_size - seq_len + 1)
        else:
            padding = []
        return sequence + padding
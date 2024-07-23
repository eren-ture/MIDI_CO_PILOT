from symusic import Score
from matplotlib import pyplot as plt
import numpy as np
import torch
import math


def tensor_to_track(notes):
    #TODO: Convert notes tensor into Symusic Track
    return None



def track_to_list(track, zero_ix=None):
    '''
    Input Symusic track\n
    Output Tensor of notes [time, duration, pitch, velocity].\n
    If zero_ix is given, all time = time - zero_ix.
    '''
    if zero_ix is not None and not isinstance(zero_ix, int):
        raise ValueError("zero_ix must be an integer or None")

    if zero_ix == None:
        notes = [[note.time, note.duration, note.pitch, note.velocity] for note in track.notes]
    else:
        notes = [[note.time-zero_ix, note.duration, note.pitch, note.velocity] for note in track.notes]
    return sorted(notes, key=lambda x: x[0])



def quantize_to_32nds(value):
    return round(value * 8) / 8

def get_clipped_tracks(midi,
                       n_bars=16,
                       zero_index=True,
                       include_empty_start=False,
                       visualize=False,
                       debug=False):
    
    # Quantize notes
    for track in midi.tracks:
        for note in track.notes:
            note.time = quantize_to_32nds(note.time)
            note.duration = quantize_to_32nds(note.duration)
    
    # Parse Time Signature
    if midi.time_signatures:
        time_sig = (midi.time_signatures[0].numerator, midi.time_signatures[0].denominator)
    else:
        time_sig = (4, 4)

    # Initialize variables
    bar_length = time_sig[0] * (4 / time_sig[1])
    if bar_length < 1:
        return []
    first_note_time = min(t.notes[0].time for t in midi.tracks if t.notes)

    if include_empty_start:
        loop_start = midi.start()
    else:
        for s in range(math.floor(midi.start()), math.ceil(midi.end()), int(bar_length)):
            if first_note_time <= s:
                loop_start = s
                break

    clipped_tracks = []

    for track in midi.tracks:

        if track.is_drum:   # Skip if drum track.
            continue

        if debug or visualize:
            print(f"Processing track \"{track.name}\"")

        clipped_track = []

        for i in range(math.floor(loop_start), math.ceil(midi.end()), int(n_bars * bar_length)):
            start_quarter = i
            end_quarter = min(start_quarter + (n_bars * bar_length), midi.end())
            
            if end_quarter - start_quarter < bar_length:  # Skip if less than one bar.
                continue
            
            clipped_part = track.clip(start_quarter, end_quarter, clip_end=True)
            if zero_index:
                notes_list = track_to_list(clipped_part, start_quarter)
            else:
                notes_list = track_to_list(clipped_part)
            clipped_track.append(notes_list)

            if debug:
                print("---")
                print(f"Clip from: {i}")
                print(f"Bars in clip: {(end_quarter - start_quarter) / bar_length}")
                print(f"Number of notes in clip: {len(clipped_part.notes)}")

            # BUG: Pianoroll doesn't work with the ttype == 'quarter'
            if visualize:
                clipped_part.ttype = 'tick'
                frame_pianoroll = clipped_part.pianoroll(['frame'])
                plt.imshow(frame_pianoroll[0,:,start_quarter:], aspect='auto')
                plt.title(f"Clip from: {i}")
                plt.show()
        
        clipped_tracks.append(clipped_track)

    return clipped_tracks
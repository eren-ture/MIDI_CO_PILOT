from symusic import Score
import json
import os
import random

def get_song_details(dir):
    """
    Input midi directory.\n
    Output song details as a dict().
    """
    try:
        midi = Score(dir)
    except:
        print(f'ERROR with: {dir}')
        return None

    # Discard songs with only one track.
    n_tracks = sum(1 if not t.is_drum else 0 for t in midi.tracks) # Count only non-drum tracks.
    if n_tracks <= 1:
        print(f'SINGLE TRACK: {dir}')
        return None

    # Discard songs with multiple time signatures.
    time_signatures = midi.time_signatures
    if len(time_signatures) > 1:
        print(f'MULTI SIGNATURE: {dir}')
        return None

    # Discard songs with multiple tempos.
    tempos = midi.tempos
    if len(tempos) > 1:
        print(f'MULTI TEMPO: {dir}')
        return None

    # Discard songs with multiple keys.
    keys = midi.key_signatures
    if len(keys) > 1:
        print(f'MULTI KEY: {dir}')
        return None

    song = {
        'Midi_Directory': dir,
        'Artist': dir.split('/')[-2],
        'Song': dir.split('/')[-1][:-4]
    }
    return song


data = []
data_folder = './archive'

for root, dirs, files in os.walk(data_folder):
    for file in files:
        if not file.endswith('.mid'):
            continue
        midi_directory = os.path.join(root, file)
        song_details = get_song_details(midi_directory)
        if song_details:
            data.append(song_details)

random.Random(42).shuffle(data)
train_percentage = 0.85
train_cutoff = int(len(data)*train_percentage)

print(f"Number of songs: {len(data)}")
print(f"{train_percentage} train data")

train_data = data[:train_cutoff]
test_data = data[train_cutoff:]

train_file_path = "data/train_data.json"
with open(train_file_path, "w") as json_file:
    json.dump(train_data, json_file, indent=4)

test_file_path = "data/test_data.json"
with open(test_file_path, "w") as json_file:
    json.dump(test_data, json_file, indent=4)
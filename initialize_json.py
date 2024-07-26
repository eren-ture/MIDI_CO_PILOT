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
        # print(f'SINGLE TRACK: {dir}')
        return None

    # Discard songs with multiple time signatures.
    time_signatures = midi.time_signatures
    if len(time_signatures) > 1:
        # print(f'MULTI SIGNATURE: {dir}')
        return None
    elif len(time_signatures) > 0:
        t_sig = (time_signatures[0].numerator, time_signatures[0].denominator)
        if t_sig[0] * (4 / t_sig[1]) < 1:
            # print(f'BAR LENGTH TOO SHORT: {dir}')
            return None
        if t_sig[0] > 16:
            print(f"RIDICULOUS TIME SIG: {dir}")
            print(f"Time Signature: {t_sig}")
            return None

    # Discard songs with multiple tempos.
    tempos = midi.tempos
    if len(tempos) > 1:
        # print(f'MULTI TEMPO: {dir}')
        return None

    # Discard songs with multiple keys.
    keys = midi.key_signatures
    if len(keys) > 1:
        # print(f'MULTI KEY: {dir}')
        return None

    song = {
        'Midi_Directory': dir,
        'Artist': dir.split(os.sep)[-2],
        'Song': dir.split(os.sep)[-1][:-4]
    }
    return song


## SET VALUES ##
data_folder = './archive'
train_percentage = 0.85

data = []

for root, dirs, files in os.walk(data_folder):
    for file in files:
        if not file.endswith('.mid'):
            continue
        midi_directory = os.path.join(root, file)
        song_details = get_song_details(midi_directory)
        if song_details:
            data.append(song_details)

random.Random(42).shuffle(data)
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
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparameters = {
    'pitch_embd_size':256,
    'velocity_embd_size':7,
    'max_quarter_notes':128,
    'num_encoder_layers':3,
    'num_decoder_layers':3,
    'num_heads':8,
    'dim_feedforward':2048,
    'dropout':0.1,
    'batch_first':True
}

max_seq_len = 512

symbols = {
    "bot" : 128,
    "eot" : 129,
    "pad" : 130
}

attn_mask_time_weight_decay = 0.2
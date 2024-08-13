import torch
import torch.nn as nn
import torch.nn.functional as F

class MidiTransformer(nn.Module):
    def __init__(
            self,
            pitch_embd_size = 256,
            velocity_embd_size = 7,
            max_quarter_notes = 128,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_heads=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        ):
        super(MidiTransformer, self).__init__()

        self.num_heads = num_heads

        self.ptch_embd_size = pitch_embd_size
        self.velo_embd_size = velocity_embd_size
        self.total_embed_size = self.ptch_embd_size + self.velo_embd_size + 1

        self.pitch_embedding = nn.Embedding(131, self.ptch_embd_size, padding_idx=130)
        self.velocity_embedding = nn.Embedding(131, self.velo_embd_size, padding_idx=130)

        # !!! IMPORTANT !!!
        # Time encoding for positional encoding.
        #   This encoding assumes the max input "n_bars" is 16 (in Dataset.py)
        #   In the dataset, this translates to 128 quarter notes.
        # If the data exceeds this length, the time embedding won't work properly.
        # This also assumes that all notes are quantized to the nearest 32nd note. (steps of 0.125 Quarter notes)
        # A nn.Linear(1, embed_size) will probably work, but I wanted to try this :)
        self.max_quarter_notes = max_quarter_notes

        self.time_pos_embedding = nn.Embedding(8 * self.max_quarter_notes + 1, self.total_embed_size)

        self.relu = F.relu

        self.softmax = F.softmax

        self.transformer = nn.Transformer(
            d_model=self.total_embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )

        self.time_ff = nn.Linear(self.total_embed_size, 1)
        self.duration_ff = nn.Linear(self.total_embed_size, 1)
        self.pitch_ff = nn.Linear(self.total_embed_size, 131)
        self.velocity_ff = nn.Linear(self.total_embed_size, 131)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask):

        B, T, _ = src.shape

        src_emb = self.embed_all(src)           # B, T, total_embed_size
        tgt_emb = self.embed_all(tgt)           # B, T, total_embed_size

        outs = self.transformer(                # B, T, total_embed_size
            src_emb,
            tgt_emb,
            src_mask=src_mask.repeat(self.num_heads, 1, 1).float(),
            tgt_mask=tgt_mask.repeat(self.num_heads, 1, 1).float(),
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_is_causal=True
        )

        out_time = self.time_ff(outs)           # B, T, 1
        out_time = self.relu(out_time)          # B, T, 1

        out_duration = self.duration_ff(outs)   # B, T, 1
        out_duration = self.relu(out_duration)  # B, T, 1

        out_pitch = self.pitch_ff(outs)         # B, T, 131
        chosen_pitch = torch.argmax(out_pitch, dim=-1, keepdim=True)

        out_velocity = self.velocity_ff(outs)   # B, T, 131
        chosen_velocity = torch.argmax(out_velocity, dim=-1, keepdim=True)

        padding_mask = ~(tgt[...,2] == 130)

        w_t, time_loss          = 1.0, F.mse_loss(out_time[padding_mask].flatten(), tgt[...,0][padding_mask])
        w_d, duration_loss      = 1.0, F.mse_loss(out_duration[padding_mask].flatten(), tgt[...,1][padding_mask])
        w_p, pitch_loss         = 1.0, F.cross_entropy(out_pitch.view(B*T, 131), tgt[...,2].view(B*T).to(torch.int64), ignore_index=130)
        w_v, velocity_loss      = 1.0, F.cross_entropy(out_velocity.view(B*T, 131), tgt[...,3].view(B*T).to(torch.int64), ignore_index=130)

        t_loss = w_t * time_loss
        d_loss = w_d * duration_loss
        p_loss = w_p * pitch_loss
        v_loss = w_v * velocity_loss
        total_loss = t_loss + d_loss + p_loss + v_loss

        concatenated_output = torch.cat([out_time, out_duration, chosen_pitch, chosen_velocity], dim=-1)
        return concatenated_output, total_loss, t_loss, d_loss, p_loss, v_loss
    
    def embed_all(self, x):
        time_input = x[...,0]
        if torch.max(time_input) >= self.max_quarter_notes:
            print("Warning: Notes exceed the max quarter notes. Check \"max_quarter_notes\" in TransformerModel.py")
        time_input = (time_input * 8).clamp(max = 8 * self.max_quarter_notes).to(int)
        pos_enc = self.time_pos_embedding(time_input)

        drtn_min, drtn_max = 0., 4.
        drtn_input = ((x[...,1] - drtn_min) / (drtn_max - drtn_min)).unsqueeze(-1)

        ptch_input = self.pitch_embedding(x[...,2].to(int))
        velo_input = self.velocity_embedding(x[...,3].to(int))

        return torch.cat([drtn_input, ptch_input, velo_input], dim=-1) + pos_enc

    def encode(self, src, src_mask, src_padding_mask):

        pos_enc = self.embed_all(src)

        return self.transformer.encoder(pos_enc, src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, tgt, memory, tgt_mask):

        pos_enc = self.embed_all(tgt)

        print("Tgt Mask Size:", tgt_mask.shape)

        tgt_mask = tgt_mask.repeat(self.num_heads, 1, 1)

        print("In Tgt Mask Size:", tgt_mask.shape)

        return self.transformer.decoder(pos_enc, memory, tgt_mask)
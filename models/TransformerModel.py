import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size,
        dropout,
        maxlen=5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class MidiTransformer(nn.Module):
    def __init__(
            self,
            embed_size,
            num_encoder_layers=6,
            num_decoder_layers=6,
            num_heads=6,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        ):
        super(MidiTransformer, self).__init__()

        self.pos_enc = PositionalEncoding(embed_size, dropout)

        self.relu = F.relu

        self.softmax = F.softmax

        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=batch_first
        )

        # self.ff = nn.Linear(embed_size, embed_size)

        self.time_ff = nn.Linear(embed_size, 1)
        self.duration_ff = nn.Linear(embed_size, 1)
        self.pitch_ff = nn.Linear(embed_size, 128)
        self.velocity_ff = nn.Linear(embed_size, 128)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, trg, src_mask, tgt_mask):

        src_emb = self.pos_enc(src)
        tgt_emb = self.pos_enc(trg)

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask
        )

        time = self.time_ff(outs)
        time = self.relu(time)

        duration = self.duration_ff(outs)
        duration = self.relu(duration)

        pitch = self.pitch_ff(outs)
        # pitch = self.softmax(pitch, dim=-1)

        velocity = self.velocity_ff(outs)
        # velocity = self.softmax(velocity, dim=-1)

        concatenated_output = torch.cat([time, duration, pitch, velocity], dim=-1)
        return concatenated_output

    def encode(self, src, src_mask):

        # embed = self.src_embedding(src)

        pos_enc = self.pos_enc(src)

        return self.transformer.encoder(pos_enc, src_mask)

    def decode(self, tgt, memory, tgt_mask):
        
        # embed = self.tgt_embedding(tgt)

        pos_enc = self.pos_enc(tgt)

        return self.transformer.decoder(pos_enc, memory, tgt_mask)
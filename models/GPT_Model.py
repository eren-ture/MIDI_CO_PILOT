import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(42)


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        input of size (batch, time-step, channels)\n
        output of size (batch, time-step, head size)
        '''
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MidiGPT(nn.Module):

    def __init__(self,
                 pitch_size,
                 velocity_size,
                 n_embed_pitch = 64,
                 n_embed_velocity = 9,
                 n_layer = 3,
                 n_head = 5,
                 block_size = 512,
                 dropout = 0.1
                 ):
        
        super().__init__()

        self.pitch_embedding = nn.Embedding(pitch_size, n_embed_pitch)
        self.velocity_embedding = nn.Embedding(velocity_size, n_embed_velocity)

        self.total_emb_size = 2 + n_embed_pitch + n_embed_velocity
        assert self.total_emb_size % n_head == 0, f"Total Embedding Size ({self.total_emb_size}) % num head ({n_head}) has to be 0."

        self.blocks = nn.Sequential(*[Block(self.total_emb_size, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(self.total_emb_size)

        self.time_head = nn.Linear(self.total_emb_size, 1)
        self.duration_head = nn.Linear(self.total_emb_size, 1)
        self.pitch_head = nn.Linear(self.total_emb_size, pitch_size)
        self.velocity_head = nn.Linear(self.total_emb_size, velocity_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, cxt, tgt=None):

        B, T, _ = cxt.shape

        x = torch.cat(      # (B, T, total_emb_size)
            [
                cxt[...,0].unsqueeze(-1),           # Float Time
                cxt[...,1].unsqueeze(-1),           # Float Duration
                self.pitch_embedding(cxt[...,2].to(torch.int64)),   # Embedded Pitch
                self.velocity_embedding(cxt[...,3].to(torch.int64)) # Embedded Velocity
            ],
            dim=-1
        )
        x = self.blocks(x)  # (B, T, total_emb_size)
        x = self.ln_f(x)    # (B, T, total_emb_size)

        out_time = self.time_head(x)            # (B, T, 1)
        out_duration = self.duration_head(x)    # (B, T, 1)
        out_pitch = self.pitch_head(x)          # (B, T, n_embed_pitch)
        out_velocity = self.velocity_head(x)    # (B, T, n_embed_velocity)

        if tgt is None:
            loss = None
        else:
            padding_mask = ~(tgt[...,2] == 129)
            n_emb_pitch = out_pitch.shape[-1]
            n_emb_velocity = out_velocity.shape[-1]

            w_t, time_loss          = 1, F.mse_loss(out_time[padding_mask].flatten(), tgt[...,0][padding_mask])
            w_d, duration_loss      = 1, F.mse_loss(out_duration[padding_mask].flatten(), tgt[...,1][padding_mask])
            w_p, pitch_loss         = 1, F.cross_entropy(out_pitch.view(B*T, n_emb_pitch), tgt[...,2].view(B*T).to(torch.int64), ignore_index=129)
            w_v, velocity_loss      = 1, F.cross_entropy(out_velocity.view(B*T, n_emb_velocity), tgt[...,3].view(B*T).to(torch.int64), ignore_index=129)

            loss = (w_t * time_loss) + (w_d * duration_loss) + (w_p * pitch_loss) + (w_v * velocity_loss)

        logits = torch.cat(
            [
                out_time,
                out_duration,
                out_pitch,
                out_velocity
            ],
            dim=-1
        )

        return logits, loss
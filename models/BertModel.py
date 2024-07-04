import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

class MidiTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dropout=0.1):
        super(MidiTransformer, self).__init__()
        
        self.config = BertConfig(
            vocab_size=256,  # just an arbitrary number
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self.model = BertModel(self.config)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, input_ids, attention_mask):
        # Flatten the input for BERT
        input_ids = input_ids.view(input_ids.size(0), -1)
        attention_mask = attention_mask.view(attention_mask.size(0), -1)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Reshape the output back to the original shape
        output = self.fc_out(last_hidden_state)
        output = output.view(-1, self.input_dim)
        return output
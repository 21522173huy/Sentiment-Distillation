
from transformers import AutoModel
from torch import nn
import torch 

class ClassifierLayer(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # Take the [CLS] token representation
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class CustomModel(nn.Module):
  def __init__(self, t5_version, num_labels, freeze_encoder = False):
    super().__init__()
    base = AutoModel.from_pretrained(t5_version)
    hidden_size = base.encoder.config.hidden_size
    # Encoder
    self.encoder = base.encoder

    # Classifier
    self.classifier = ClassifierLayer(hidden_size, num_labels)
    if freeze_encoder:
      self.freeze_encoder_fn()

  def freeze_encoder_fn(self):
    for param in self.encoder.parameters():
        param.requires_grad = False

    # Ensure the classifier layer's parameters are not frozen
    for param in self.classifier.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    output = self.encoder(input_ids = input_ids,
                          attention_mask = attention_mask).last_hidden_state

    return self.classifier(output)

class CustomT5_FromLarge(nn.Module):
  def __init__(self, num_labels, num_blocks = 9, t5_version = 'VietAI/vit5-large', freeze_encoder = False):
    super().__init__()
    base = AutoModel.from_pretrained(t5_version).encoder
    hidden_size = base.config.hidden_size
    # Embedding
    self.embed_tokens = base.embed_tokens
    # Encoder
    self.blocks = base.block[:num_blocks]
    # Other Component
    self.final_layer_norm = base.final_layer_norm
    self.dropout = base.final_layer_norm

    # Classifier
    self.classifier = ClassifierLayer(hidden_size, num_labels)
    if freeze_encoder:
      self.freeze_encoder_fn()

  def freeze_encoder_fn(self):
    for param in self.encoder.parameters():
        param.requires_grad = False

    # Ensure the classifier layer's parameters are not frozen
    for param in self.classifier.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    embedding_output = self.embed_tokens(input_ids)

    if attention_mask is not None:
        # Reshape the attention mask to match the shape required for multi-head attention
        attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_length)

    for layer in self.blocks:
        embedding_output = layer(embedding_output, attention_mask=attention_mask)[0]

    hidden_states = self.final_layer_norm(embedding_output)
    hidden_states = self.dropout(hidden_states)
    return self.classifier(hidden_states)

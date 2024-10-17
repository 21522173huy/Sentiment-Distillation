
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Callable, Any, Tuple
from torch import nn

class TeacherModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super(TeacherModel, self).__init__()
        self.teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.trainable = True
        self.load_in_nbits = 16  # Adjust as needed

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.teacher_model.to(self.device)
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

    def save_checkpoint(self, save_path: str):
        self.teacher_model.save_pretrained(save_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.logits

class CustomRoberta_FromLarge(nn.Module):
  def __init__(self, num_labels, num_blocks = 9, roberta_version = 'FacebookAI/roberta-large', freeze_encoder = False):
    super().__init__()

    base = AutoModelForSequenceClassification.from_pretrained(roberta_version, 
                                                              num_labels = num_labels)
    hidden_size = base.config.hidden_size

    # Embedding
    self.embeddings = base.roberta.embeddings

    # Encoder
    self.layers = base.roberta.encoder.layer[:num_blocks]

    # Classifier
    self.classifier = base.classifier
    if freeze_encoder:
      self.freeze_encoder_fn()

  def freeze_encoder_fn(self):
    for param in self.encoder.parameters():
        param.requires_grad = False

    # Ensure the classifier layer's parameters are not frozen
    for param in self.classifier.parameters():
        param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    embedding_output = self.embeddings(input_ids)

    if attention_mask is not None:
        # Reshape the attention mask to match the shape required for multi-head attention
        attention_mask = attention_mask[:, None, None, :]  # (batch_size, 1, 1, seq_length)

    for layer in self.layers:
        embedding_output = layer(embedding_output, attention_mask=attention_mask)[0]

    return self.classifier(embedding_output)

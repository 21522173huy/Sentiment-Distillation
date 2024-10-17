import torch
from torch import nn
import argparse
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['ViT5', 'Roberta-XLM'], type=str, required=True)
    parser.add_argument('--teacher_or_student', choices=['teacher', 'student'],  type=str, required=True)
    parser.add_argument('--model_type', choices=['large', 'base'],  type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--input_sentence', type=str, required=True)

    args = parser.parse_args()

    if args.model_name == 'ViT5':
      from model.t5_model import CustomModel

      if args.teacher_or_student == 'teacher':
        # Teacher
        model = CustomModel(t5_version = 'VietAI/vit5-large', num_labels = 3)

      # Student
      elif args.teacher_or_student == 'student':
        if args.model_type == 'base':
          model = CustomModel(t5_version = 'VietAI/vit5-base', num_labels = 3)
        elif args.model_type == 'large':
          from model.t5_model import CustomT5_FromLarge
          model = CustomT5_FromLarge(num_labels = 3, num_blocks = 6)

      # Tokenizer
      tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-large')

    elif args.model_name == 'Roberta-XLM':
      from model.roberta_model import TeacherModel

      # Teacher
      if args.teacher_or_student == 'teacher':
        model = TeacherModel(model_name = 'FacebookAI/xlm-roberta-large', num_labels = 3)

      elif args.teacher_or_student == 'student':
        model = TeacherModel(model_name = 'FacebookAI/xlm-roberta-base', num_labels = 3)
      # Tokenizer
      tokenizer = model.tokenizer

    # Load Checkpoint
    model.load_state_dict(torch.load(args.model_checkpoint, map_location = torch.device('cuda')))

    # Prediction
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenized_sentence = tokenizer(args.input_sentence, return_tensors = 'pt')
    model.eval(), model.to(device)
    with torch.no_grad():
      prediction = model(input_ids = tokenized_sentence['input_ids'].to(device), attention_mask = tokenized_sentence['attention_mask'].to(device))
      output_logits = prediction.softmax(dim=-1)[0]

    result = {
        'Negative': output_logits[0].item(),
        'Positive': output_logits[1].item(),
        'Neutral': output_logits[2].item()
    }

    print(f'For Sentence: "{args.input_sentence}"')
    print("Result is:")
    for sentiment, score in result.items():
        print(f"  {sentiment}: {score:.4f}")

if __name__ == "__main__":
    main()

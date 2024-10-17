
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import yaml
import sys
import argparse
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch

def evaluate_model(model, test_dataloader, average = 'micro', save_score_path='evaluation_results.json', save_prediction_path='Teacher-Student-Prediction.json'):
    model.eval(), model.to('cuda')

    teacher_predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids, attention_mask = batch['input_ids'].to('cuda'), batch['attention_mask'].to('cuda')
            labels = batch['labels'].to('cuda')

            # Predictions from the teacher model
            teacher_outputs = model(input_ids, attention_mask=attention_mask)
            teacher_preds = torch.argmax(teacher_outputs, dim=-1)
            teacher_predictions.extend(teacher_preds.cpu().numpy().tolist())

            # True labels
            true_labels.extend(labels.cpu().numpy().tolist())

    # Calculate metrics for teacher model
    teacher_accuracy = accuracy_score(true_labels, teacher_predictions)
    teacher_precision = precision_score(true_labels, teacher_predictions, average=average)
    teacher_recall = recall_score(true_labels, teacher_predictions, average=average)
    teacher_f1 = f1_score(true_labels, teacher_predictions, average=average)

    # Save results to a JSON file
    results = {
      "accuracy_score": teacher_accuracy,
      "precision_score": teacher_precision,
      "recall_score": teacher_recall,
      "f1_score": teacher_f1
    }

    # import json
    # with open('results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

    return results, teacher_predictions, true_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['ViT5', 'Roberta'], type=str, required=True)
    parser.add_argument('--teacher_or_student', choices=['teacher', 'student'],  type=str, required=True)
    parser.add_argument('--model_type', choices=['large', 'base'],  type=str, required=True)
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--language', choices=['vietnamese', 'english'], type=str, default = 'vietnamese')
    parser.add_argument('--full_test', type=bool, default=False)
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

    elif args.model_name == 'Roberta':
      from model.roberta_model import TeacherModel

      # Teacher
      if args.teacher_or_student == 'teacher':
        model = TeacherModel(model_name = 'FacebookAI/roberta-large', num_labels = 3)

      elif args.teacher_or_student == 'student':
        model = TeacherModel(model_name = 'FacebookAI/roberta-base', num_labels = 3)
      # Tokenizer
      tokenizer = model.tokenizer

    # Load Checkpoint
    model.load_state_dict(torch.load(args.model_checkpoint, map_location = torch.device('cuda')))
    # Load the dataset
    from dataset import create_dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer = tokenizer,
                                                                           batch_size=args.batch_size,
                                                                           rdrsegmenter = None,
                                                                           language = args.language,
                                                                           full_test = args.full_test) # rdrsegmenter is used for PhoBert


    sample = next(iter(test_dataloader))
    print(tokenizer.decode(sample.input_ids[0], skip_special_tokens = True))
    
    results, _, _ = evaluate_model(model, test_dataloader)
    print(f'Result on Test Set: {results}')

    # # Save results to a JSON file
    # import json
    # id = args.teacher_name.split('/')[-1]
    # with open(f'{id}_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

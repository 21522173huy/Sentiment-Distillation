import argparse
import torch
from torch import nn
from transformers import AutoTokenizer

# Move to parent folder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set manual seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

folder_path = 'teacher/checkpoints'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")

from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            lr_factor = float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_factor = max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
        
        # Print the current step and learning rate factor
        print(f"Step: {current_step}, Learning Rate Factor: {lr_factor}")
        
        return lr_factor
    return LambdaLR(optimizer, lr_lambda)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher_name', choices=['T5', 'Roberta'], type=str, required=True)
    parser.add_argument('--language', choices=['vietnamese', 'english'], type=str, default = 'vietnamese')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    args = parser.parse_args()

    num_labels = 1

    if args.teacher_name == 'T5':
      from model.t5_model import CustomModel

      print(f'USE T5 MODEL')
        
      if args.language == 'vietnamese': 
          t5_version = 'VietAI/vit5-large'
          num_labels = 3
      elif args.language == 'english' : 
          t5_version = 'google/flan-t5-large'
          num_labels = 2
        
      teacher_model = CustomModel(t5_version = t5_version, num_labels = num_labels)
      tokenizer = AutoTokenizer.from_pretrained(t5_version)
      optimizer = torch.optim.AdamW(teacher_model.parameters(), weight_decay=0.01, lr = 2e-05)

    elif args.teacher_name == 'Roberta':

      if args.language == 'vietnamese' : num_labels = 3
      else : num_labels = 2
          
      print('Num Labels: ', num_labels)
      from model.roberta_model import TeacherModel

      print(f'USE ROBERTA-XLM MODEL')
      teacher_model = TeacherModel(model_name = 'FacebookAI/roberta-large', num_labels = num_labels)
      tokenizer = teacher_model.tokenizer
      optimizer = torch.optim.Adam(params=teacher_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # Load the dataset
    from dataset import create_dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer = tokenizer,
                                                                           batch_size=args.batch_size,
                                                                           language = args.language,
                                                                           rdrsegmenter = None) # rdrsegmenter is used for PhoBert

    # Check whether English version is used correctly
    sample = next(iter(train_dataloader))
    print(tokenizer.decode(sample.input_ids[0], skip_special_tokens = True))

    # Finetuning Config
    criterion = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Scheduler with warm-up and linear decay
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Finetuning
    from teacher.finetune_function import finetune_teacher
    train_loss, test_loss, train_metrics, test_metrics = finetune_teacher(teacher_model=teacher_model,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          scheduler=scheduler,
                                                                          checkpoint_path = f'teacher/checkpoints/{args.teacher_name}-Teacher-best.pt',
                                                                          result_path = f'teacher/checkpoints/{args.teacher_name}-Teacher-results.json',
                                                                          max_grad_norm=1.0,
                                                                          epochs=args.epochs,
                                                                          patience=5)

    results = {
        'Train': {
            'Loss': train_loss,
            'Metrics': train_metrics,
        },
        'Val': {
            'Loss': test_loss,
            'Metrics': test_metrics,
        }
    }

    # # Save results to a JSON file
    # import json
    # id = args.teacher_name.split('/')[-1]
    # with open(f'{id}_results.json', 'w') as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

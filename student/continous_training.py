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

folder_path = 'student/checkpoints'
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
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--student_type', choices=['large', 'base'], type=str, default='base')
    parser.add_argument('--language', choices=['vietnamese', 'english'], type=str, default='vietnamese')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-04)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--soft_weight', type=float)
    parser.add_argument('--hard_weight', type=float)
    parser.add_argument('--student_checkpoint', type=str, default=None)  # New argument
    args = parser.parse_args()

    num_labels = 1
    if args.teacher_name == 'T5':
        from model.t5_model import CustomModel

        if args.language == 'vietnamese':
            large_t5_version = 'VietAI/vit5-large'
            base_t5_version = 'VietAI/vit5-base'
            num_labels = 3

        elif args.language == 'english':
            large_t5_version = 'google/flan-t5-large'
            base_t5_version = 'google/flan-t5-base'
            num_labels = 2

        print('Language: ', args.language)
        print('Num Labels: ', num_labels)

        # Teacher
        teacher_model = CustomModel(t5_version=large_t5_version, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(large_t5_version)

        # Student
        if args.student_type == 'base':
            student_model = CustomModel(t5_version=base_t5_version, num_labels=num_labels)
        elif args.student_type == 'large':
            from model.t5_model import CustomT5_FromLarge
            student_model = CustomT5_FromLarge(num_labels=num_labels, num_blocks=6, t5_version=large_t5_version)

        # Load student checkpoint if provided
        if args.student_checkpoint is not None:
            student_model.load_state_dict(torch.load(args.student_checkpoint))
            print(f"Loaded student checkpoint from {args.student_checkpoint}")

        # Optimizer
        optimizer = torch.optim.AdamW(student_model.parameters(), weight_decay=0.01, lr=2e-05)

    elif args.teacher_name == 'Roberta':

        if args.language == 'vietnamese':
            num_labels = 3
        else:
            num_labels = 2

        print('Num Labels: ', num_labels)

        from model.roberta_model import TeacherModel

        # Roberta Version
        roberta_large = 'FacebookAI/roberta-large'
        roberta_base = 'FacebookAI/roberta-base'

        # Teacher
        teacher_model = TeacherModel(model_name=roberta_large, num_labels=num_labels)
        tokenizer = teacher_model.tokenizer

        # Student
        if args.student_type == 'base':
            student_model = TeacherModel(model_name=roberta_base, num_labels=num_labels)

        elif args.student_type == 'large':
            from model.roberta_model import CustomRoberta_FromLarge
            student_model = CustomRoberta_FromLarge(num_labels=num_labels, num_blocks=6, roberta_version=roberta_large)

        # Load student checkpoint if provided
        if args.student_checkpoint is not None:
            student_model.load_state_dict(torch.load(args.student_checkpoint))
            print(f"Loaded student checkpoint from {args.student_checkpoint}")

        # Optimizer
        optimizer = torch.optim.Adam(params=student_model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)

    # Load the dataset
    from dataset import create_dataloaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(tokenizer=tokenizer,
                                                                           batch_size=args.batch_size,
                                                                           language=args.language,
                                                                           rdrsegmenter=None)  # rdrsegmenter is used for PhoBert

    # Check whether English version is used correctly
    sample = next(iter(train_dataloader))
    print(tokenizer.decode(sample.input_ids[0], skip_special_tokens=True))

    # Finetuning Config
    criterion = nn.CrossEntropyLoss()

    # Scheduler with warm-up and linear decay
    num_training_steps = args.epochs * len(train_dataloader)
    num_warmup_steps = 10000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Finetuning
    type_in_path = 'Base' if args.student_type == 'base' else 'Large'
    if args.student_type == 'Roberta-XLM':
        type_in_path = 'Base'

    from student.training_function import training_student
    train_loss, test_loss, train_metrics, test_metrics = training_student(student_model=student_model,
                                                                          teacher_model=teacher_model,
                                                                          train_dataloader=train_dataloader,
                                                                          val_dataloader=val_dataloader,
                                                                          test_dataloader=test_dataloader,
                                                                          optimizer=optimizer,
                                                                          criterion=criterion,
                                                                          scheduler=scheduler,
                                                                          max_grad_norm=1.0,
                                                                          epochs=args.epochs,
                                                                          save_path=f'student/checkpoints/{type_in_path}-Student-{args.teacher_name}-{int(args.soft_weight * 100)}{int(args.hard_weight * 100)}-best.pt',
                                                                          patience=5,
                                                                          temperature=2,
                                                                          soft_weight=args.soft_weight,
                                                                          hard_weight=args.hard_weight)
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

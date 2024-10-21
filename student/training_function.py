
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation import evaluate_model
import torch
from torch import nn
from early_stopping import EarlyStopping

def calculate_soft_loss(student_logits, teacher_logits, temperature):
    """
    Calculate the soft loss using KL divergence.
    """
    soft_labels = nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_loss = nn.functional.kl_div(
        nn.functional.log_softmax(student_logits / temperature, dim=-1),
        soft_labels,
        reduction='batchmean'
    ) * (temperature ** 2)
    return soft_loss

def step(student_model, teacher_model, dataloader, optimizer, criterion, device, temperature=2.0, soft_weight=0.5, hard_weight=0.5, max_grad_norm=1.0, mode='Train', average='micro'):
    teacher_model.eval()
    if mode == 'Train': student_model.train()
    else: student_model.eval()

    total_loss = 0
    metrics_epoch = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].float().to(device)
        labels = batch['labels'].to(device)

        if mode == 'Train': optimizer.zero_grad()

        with torch.set_grad_enabled(mode == 'Train'):
            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate losses
            soft_loss = calculate_soft_loss(student_logits, teacher_logits, temperature)
            hard_loss = criterion(student_logits, labels)
            loss = soft_weight * soft_loss + hard_weight * hard_loss

            # Backward
            if mode == 'Train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_grad_norm)
                optimizer.step()

        # Loss and Metrics
        total_loss += loss.item()

        preds = student_logits.argmax(dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average=average, zero_division=0)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        metrics_epoch['acc'] += acc
        metrics_epoch['precision'] += precision
        metrics_epoch['recall'] += recall
        metrics_epoch['f1'] += f1

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_epoch.items()}

    return avg_loss, avg_metrics

def training_student(student_model, teacher_model,
                     train_dataloader, val_dataloader, test_dataloader,
                     optimizer, criterion, scheduler, epochs,
                     checkpoint_path = 'checkpoint_best.pt', result_path = 'results.json',
                     max_grad_norm=1.0, patience=5,
                     temperature=2.0, soft_weight=0.5, hard_weight=0.5):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model.to(device)
    student_model.to(device)

    train_losses, test_losses = [], []
    train_metrics_list, test_metrics_list = [], []
    best_checkpoint_path = None
    best_f1_score = float(-1)

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    for epoch in range(epochs):
        # Loss and Metrics
        print("Training")
        train_loss, train_metrics = step(student_model, teacher_model, train_dataloader, optimizer, criterion, device, temperature, soft_weight, hard_weight, max_grad_norm, mode='Train')
        print("Validating")
        val_loss, val_metrics = step(student_model, teacher_model, val_dataloader, optimizer, criterion, device, temperature, soft_weight, hard_weight, max_grad_norm, mode='Val')
        print('Testing')
        results, y_pred, y_true = evaluate_model(student_model, test_dataloader, average = 'micro')

        # Obtain Loss and Metrics
        train_losses.append(train_loss)
        train_metrics_list.append(train_metrics)
        test_losses.append(val_loss)
        test_metrics_list.append(val_metrics)

        # Step the scheduler
        # scheduler.step(val_loss)
        scheduler.step()

        # Early Stopping
        early_stopping(val_loss, student_model)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics['acc']:.4f}, "
              f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}, "
              f"Train F1: {train_metrics['f1']:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_metrics['acc']:.4f}, "
              f"Validation Precision: {val_metrics['precision']:.4f}, Validation Recall: {val_metrics['recall']:.4f}, "
              f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Results in Test: {results}\n")

        # Save best F1-Score checkpoint
        if results['f1_score'] > best_f1_score:
            best_f1_score = results['f1_score']
            torch.save({
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            import json
            with open(result_path, 'w') as f:
                json.dump({
                    'results': results,
                    'y_true': y_true,
                    'y_pred': y_pred,
                }, f, indent=4)

        # Stop
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_losses, test_losses, train_metrics_list, test_metrics_list

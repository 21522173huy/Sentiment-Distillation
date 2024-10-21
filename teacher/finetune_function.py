
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from evaluation import evaluate_model
import torch
from early_stopping import EarlyStopping

def step(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0, mode='Train', average='micro'):
    if mode == 'Train': model.train()
    else: model.eval()

    total_loss = 0
    metrics_epoch = {'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    for batch in tqdm(dataloader):
        input_ids, attention_mask = batch['input_ids'].to(device), batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.set_grad_enabled(mode == 'Train'):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output, labels)

            # Backward
            if mode == 'Train':
                optimizer.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Loss and Metrics
        total_loss += loss.item()

        preds = output.argmax(dim=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels.cpu(), preds.cpu(), average=average, zero_division=0)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        metrics_epoch['acc'] += acc
        metrics_epoch['precision'] += precision
        metrics_epoch['recall'] += recall
        metrics_epoch['f1'] += f1

    avg_loss = total_loss / len(dataloader)
    avg_metrics = {k: v / len(dataloader) for k, v in metrics_epoch.items()}

    return avg_loss, avg_metrics

def finetune_teacher(teacher_model, 
                     train_dataloader, val_dataloader, test_dataloader, 
                     optimizer, criterion, scheduler, epochs, 
                     checkpoint_path, result_path,
                     max_grad_norm=1.0, patience=5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    teacher_model.to(device)

    train_losses, test_losses = [], []
    train_metrics_list, test_metrics_list = [], []

    best_f1_score = float(0)
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    for epoch in range(epochs):
        # Loss and Metrics
        print("Training")
        train_loss, train_metrics = step(teacher_model, train_dataloader, optimizer, criterion, device, max_grad_norm, mode='Train')
        print("Validating")
        val_loss, val_metrics = step(teacher_model, val_dataloader, optimizer, criterion, device, max_grad_norm, mode='Val')
        print('Testing')
        results, y_true, y_pred = evaluate_model(teacher_model, test_dataloader, average = 'micro')

        # Obtain Loss and Metrics
        train_losses.append(train_loss)
        train_metrics_list.append(train_metrics)
        test_losses.append(val_loss)
        test_metrics_list.append(val_metrics)

        # Step the scheduler
        # scheduler.step(val_loss)
        scheduler.step()

        # Early Stopping
        early_stopping(val_loss, teacher_model)

        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_metrics['acc']:.4f}, "
              f"Train Precision: {train_metrics['precision']:.4f}, Train Recall: {train_metrics['recall']:.4f}, "
              f"Train F1: {train_metrics['f1']:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_metrics['acc']:.4f}, "
              f"Validation Precision: {val_metrics['precision']:.4f}, Validation Recall: {val_metrics['recall']:.4f}, "
              f"Validation F1: {val_metrics['f1']:.4f}")
        print(f"Results in Test: {results}\n")

        # Save Best F1-score checkpoint
        if results['f1_score'] > best_f1_score:
            best_f1_score = results['f1_score']
            torch.save({
                'model_state_dict': teacher_model.state_dict(),
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

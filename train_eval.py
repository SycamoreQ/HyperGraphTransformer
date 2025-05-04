import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import numpy as np
from einops import rearrange, repeat
import os

class_names = ['yes', 'no']

def save_checkpoint(model, optimizer, scheduler, epoch, history, best_val_acc, filename):
    """
    Save model checkpoint to file
    
    Args:
        model: The model to save
        optimizer: The optimizer used for training
        scheduler: The learning rate scheduler (if any)
        epoch: Current epoch number
        history: Dictionary containing training/validation metrics
        best_val_acc: Best validation accuracy achieved so far
        filename: Path to save the checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history,
        'best_val_acc': best_val_acc
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    """
    Load model checkpoint from file
    
    Args:
        model: The model to load weights into
        optimizer: The optimizer to load state into
        scheduler: The learning rate scheduler to load state into
        filename: Path to the checkpoint file
        
    Returns:
        model: Model with loaded weights
        optimizer: Optimizer with loaded state
        scheduler: Scheduler with loaded state
        start_epoch: Epoch to resume from
        history: Dictionary containing training/validation metrics
        best_val_acc: Best validation accuracy achieved so far
    """
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint['history']
        best_val_acc = checkpoint['best_val_acc']
        
        return model, optimizer, scheduler, start_epoch, history, best_val_acc
    else:
        print(f"No checkpoint found at {filename}")
        return model, optimizer, scheduler, 0, {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}, 0.0

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, 
                num_epochs=25, device='cuda', resume=False, checkpoint_dir='checkpoints', 
                checkpoint_interval=1):
    """
    Train the model with checkpointing support
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        resume: Whether to resume training from a checkpoint
        checkpoint_dir: Directory to save checkpoints
        checkpoint_interval: How often to save regular checkpoints (in epochs)
        
    Returns:
        model: Trained model
        history: Dictionary containing training/validation metrics
    """
    # Set up checkpoint paths
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
    
    # Initialize or load history and best accuracy
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    start_epoch = 0
    
    # Resume from checkpoint if requested
    if resume and os.path.exists(checkpoint_path):
        model, optimizer, scheduler, start_epoch, history, best_val_acc = load_checkpoint(
            model, optimizer, scheduler, checkpoint_path)
        print(f"Resuming training from epoch {start_epoch}")
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Track stats
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate average loss and accuracy for this epoch
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track stats
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.sampler)
        val_acc = val_correct / val_total
        
        # Update learning rate if scheduler is provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save checkpoint at regular intervals
        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                history, best_val_acc, checkpoint_path
            )
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                history, best_val_acc, best_model_path
            )
            print(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')
    save_checkpoint(
        model, optimizer, scheduler, num_epochs-1, 
        history, best_val_acc, final_checkpoint_path
    )
    
    # Load best model
    model, _, _, _, _, _ = load_checkpoint(model, None, None, best_model_path)
    return model, history

def evaluate_model(model, test_loader, criterion=None, device='cuda'):
    """
    Evaluate model performance on a test dataset
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function (optional)
        device: Device to evaluate on ('cuda' or 'cpu')
        
    Returns:
        accuracy: Overall accuracy on the test set
        precision: Precision score
        recall: Recall score
        f1: F1 score
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    if criterion is not None:
        avg_loss = test_loss / len(test_loader.sampler)
        print(f"Test Loss: {avg_loss:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy, precision, recall, f1

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training/validation metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")

def load_best_model(model_class, model_path, **model_kwargs):
    """
    Load a trained model from a checkpoint file
    
    Args:
        model_class: The model class to instantiate
        model_path: Path to the checkpoint file
        model_kwargs: Keyword arguments for model initialization
        
    Returns:
        model: Loaded model with weights from checkpoint
    """
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    return model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
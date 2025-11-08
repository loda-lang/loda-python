"""
Training script for the LODA LLM (Large Language Model).

This script handles the complete training pipeline:
1. Load and preprocess training data
2. Set up the model and training loop
3. Train the model with proper validation
4. Save the trained model
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import List, Dict, Optional
import argparse
from tqdm import tqdm

from .data_preprocessing import DataPreprocessor, TrainingExample
from .model import LodaT5Model


class LodaDataset(Dataset):
    """PyTorch dataset for LODA training examples."""
    
    def __init__(self, examples: List[TrainingExample], model: LodaT5Model, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            examples: List of training examples
            model: LodaT5Model instance for tokenization
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.model = model
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Prepare input (description)
        input_encoding = self.model.prepare_input([example.description])
        
        # Prepare target (LODA code)
        target_encoding = self.model.prepare_target([example.loda_code])
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze()
        }


class LodaTrainer:
    """Trainer class for LODA LLM."""
    
    def __init__(self, 
                 model: LodaT5Model,
                 train_dataset: LodaDataset,
                 val_dataset: Optional[LodaDataset] = None,
                 learning_rate: float = 5e-5,
                 batch_size: int = 8,
                 num_epochs: int = 3,
                 warmup_steps: int = 500,
                 save_dir: str = "loda_llm_model"):
        """
        Initialize the trainer.
        
        Args:
            model: LodaT5Model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            learning_rate: Learning rate
            batch_size: Batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate schedule
            save_dir: Directory to save the model
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.model.to(self.device)
        
        # Set up data loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                collate_fn=self._collate_fn
            )
        
        # Set up optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Set up learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        # Pad sequences to the same length
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        decoder_attention_masks = [item['decoder_attention_mask'] for item in batch]
        
        # Pad input sequences
        max_input_len = max(len(seq) for seq in input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        
        for i in range(len(input_ids)):
            seq_len = len(input_ids[i])
            pad_len = max_input_len - seq_len
            
            padded_input_ids.append(
                torch.cat([input_ids[i], torch.zeros(pad_len, dtype=torch.long)])
            )
            padded_attention_masks.append(
                torch.cat([attention_masks[i], torch.zeros(pad_len, dtype=torch.long)])
            )
        
        # Pad target sequences
        max_target_len = max(len(seq) for seq in labels)
        padded_labels = []
        padded_decoder_masks = []
        
        for i in range(len(labels)):
            seq_len = len(labels[i])
            pad_len = max_target_len - seq_len
            
            # For labels, use -100 for padding (ignored in loss calculation)
            padded_labels.append(
                torch.cat([labels[i], torch.full((pad_len,), -100, dtype=torch.long)])
            )
            padded_decoder_masks.append(
                torch.cat([decoder_attention_masks[i], torch.zeros(pad_len, dtype=torch.long)])
            )
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': torch.stack(padded_labels),
            'decoder_attention_mask': torch.stack(padded_decoder_masks)
        }
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model.forward(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model."""
        if not self.val_dataset:
            return None
        
        self.model.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for batch in progress_bar:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                progress_bar.set_postfix({'val_loss': loss.item()})
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Train the model."""
        print(f"Training on device: {self.device}")
        print(f"Training examples: {len(self.train_dataset)}")
        if self.val_dataset:
            print(f"Validation examples: {len(self.val_dataset)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"{self.save_dir}_best")
                    print("Saved best model")
            
            # Save checkpoint
            self.save_model(f"{self.save_dir}_epoch_{epoch + 1}")
        
        print("\nTraining completed!")
        return self.model
    
    def save_model(self, path: str):
        """Save the model."""
        self.model.save_model(path)


def train_loda_llm(programs_dir: str,
                   output_dir: str = "loda_llm_model",
                   model_name: str = "t5-small",
                   max_examples: int = -1,
                   val_split: float = 0.1,
                   batch_size: int = 8,
                   learning_rate: float = 5e-5,
                   num_epochs: int = 3):
    """
    Main training function.
    
    Args:
        programs_dir: Directory containing OEIS programs
        output_dir: Directory to save the trained model
        model_name: Base T5 model to use
        max_examples: Maximum number of training examples (-1 for all)
        val_split: Fraction of data to use for validation
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of training epochs
    """
    print("Preparing training data...")
    
    # Create training examples
    preprocessor = DataPreprocessor(programs_dir)
    examples = preprocessor.create_training_examples(max_examples)
    
    if len(examples) == 0:
        print("No training examples found!")
        return None
    
    # Augment examples
    print("Augmenting training examples...")
    examples = preprocessor.augment_descriptions(examples)
    
    # Split into train/validation
    if val_split > 0:
        split_idx = int(len(examples) * (1 - val_split))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
    else:
        train_examples = examples
        val_examples = None
    
    print(f"Training examples: {len(train_examples)}")
    if val_examples:
        print(f"Validation examples: {len(val_examples)}")
    
    # Create model
    print(f"Creating model based on {model_name}...")
    model = LodaT5Model(model_name)
    
    # Create datasets
    train_dataset = LodaDataset(train_examples, model)
    val_dataset = LodaDataset(val_examples, model) if val_examples else None
    
    # Create trainer
    trainer = LodaTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        save_dir=output_dir
    )
    
    # Train the model
    trained_model = trainer.train()
    
    # Save final model
    trained_model.save_model(output_dir)
    print(f"Final model saved to {output_dir}")
    
    return trained_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LODA LLM")
    parser.add_argument("--programs_dir", type=str, required=True,
                        help="Directory containing OEIS programs")
    parser.add_argument("--output_dir", type=str, default="loda_llm_model",
                        help="Output directory for trained model")
    parser.add_argument("--model_name", type=str, default="t5-small",
                        help="Base T5 model to use")
    parser.add_argument("--max_examples", type=int, default=-1,
                        help="Maximum number of training examples (-1 for all)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    train_loda_llm(
        programs_dir=args.programs_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
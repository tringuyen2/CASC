import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import os
import sys
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.casc_model import CASCModel
from data.dataset import PersonSearchDataset, collate_fn


def setup_logging(log_dir='logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'train_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def train_one_epoch(model, dataloader, optimizer, device, epoch, logger):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    loss_dict = {
        'itc': 0, 'icc': 0, 'tcc': 0, 'itm': 0, 'lm': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, batch in enumerate(pbar):
        try:
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(images, text_ids, attention_mask, labels)
            loss = outputs['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            for key in loss_dict.keys():
                loss_dict[key] += outputs[f'loss_{key}'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'itc': f'{outputs["loss_itc"].item():.4f}',
                'W_i': f'{outputs["W_i"].item():.3f}',
                'W_t': f'{outputs["W_t"].item():.3f}'
            })
            
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            continue
    
    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_dict.keys():
        loss_dict[key] /= num_batches
    
    return avg_loss, loss_dict


@torch.no_grad()
def validate(model, dataloader, device):
    """Validation function"""
    model.eval()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        try:
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images, text_ids, attention_mask, labels)
            total_loss += outputs['total_loss'].item()
            
        except Exception as e:
            print(f"Validation error: {e}")
            continue
    
    return total_loss / len(dataloader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    torch.save(checkpoint, path)
    print(f"✓ Saved checkpoint: {path}")


def main():
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting CASC training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    logger.info("Creating model...")
    try:
        model = CASCModel(config).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model created with {num_params:,} parameters")
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.info("Trying to download pretrained models...")
        # This will trigger download of required models
        from transformers import CLIPVisionModel, BertModel
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        BertModel.from_pretrained('bert-base-uncased')
        model = CASCModel(config).to(device)
    
    # Create dataset and dataloader
    logger.info(f"Loading dataset from {config.data_root}...")
    try:
        train_dataset = PersonSearchDataset(
            data_root=config.data_root,
            split='train',
            image_size=config.image_size
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )
    
    # Training loop
    best_loss = float('inf')
    logger.info("Starting training loop...")
    
    for epoch in range(1, config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )
        
        # Log results
        logger.info(f"Epoch {epoch} Results:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  ITC Loss: {loss_dict['itc']:.4f}")
        logger.info(f"  ICC Loss: {loss_dict['icc']:.4f}")
        logger.info(f"  TCC Loss: {loss_dict['tcc']:.4f}")
        logger.info(f"  ITM Loss: {loss_dict['itm']:.4f}")
        logger.info(f"  LM Loss: {loss_dict['lm']:.4f}")
        
        # Update learning rate
        scheduler.step()
        logger.info(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                'checkpoints/best_model.pth'
            )
            logger.info(f"  ✓ New best model saved! Loss: {avg_loss:.4f}")
        
        # Save regular checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                f'checkpoints/checkpoint_epoch_{epoch}.pth'
            )
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    main()

    
"""
Updated Training script for CASC with custom image-text pair dataset
"""
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
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config


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


def load_dataset(data_root, split, batch_size):
    """Load dataset - try custom format first, then fallback to original"""
    
    # Try custom image-text pair dataset first
    try:
        from data.custom_dataset import ImageTextPairDataset, collate_fn
        
        dataset = ImageTextPairDataset(
            data_root=data_root,
            split=split,
            image_size=config.image_size
        )
        
        print(f"✓ Using custom image-text pair dataset")
        
    except Exception as e:
        print(f"Custom dataset failed: {e}")
        print(f"Trying original dataset format...")
        
        try:
            from data.dataset import PersonSearchDataset, collate_fn
            
            dataset = PersonSearchDataset(
                data_root=data_root,
                split=split,
                image_size=config.image_size
            )
            
            print(f"✓ Using original dataset format")
            
        except Exception as e2:
            print(f"Original dataset also failed: {e2}")
            raise ValueError("Could not load dataset with any format!")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    
    return dataset, dataloader


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
            
            # Check for NaN
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}, skipping...")
                continue
            
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
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate averages
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key] / num_batches if num_batches > 0 else 0
    
    return avg_loss, loss_dict


@torch.no_grad()
def validate(model, dataloader, device, logger):
    """Validation function"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        try:
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(images, text_ids, attention_mask, labels)
            total_loss += outputs['total_loss'].item()
            num_batches += 1
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0


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


def main(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging()
    logger.info("="*60)
    logger.info("Starting CASC training...")
    logger.info("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create model
    logger.info("\nCreating model...")
    try:
        from models.casc_model import CASCModel
        
        model = CASCModel(config).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created with {num_params:,} parameters")
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.info("Downloading pretrained models...")
        
        # Download required models
        from transformers import CLIPVisionModel, BertModel, BertTokenizer
        
        logger.info("Downloading CLIP...")
        CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        
        logger.info("Downloading BERT...")
        BertModel.from_pretrained('bert-base-uncased')
        BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Try again
        from models.casc_model import CASCModel
        model = CASCModel(config).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model created with {num_params:,} parameters")
    
    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"✓ Resumed from epoch {checkpoint['epoch']}")
        else:
            logger.warning(f"Checkpoint {args.resume} not found, starting from scratch")
    
    # Create dataset and dataloader
    logger.info(f"\nLoading dataset from {config.data_root}...")
    try:
        train_dataset, train_loader = load_dataset(
            data_root=config.data_root,
            split='train',
            batch_size=config.batch_size
        )
        
        logger.info(f"✓ Training dataset size: {len(train_dataset)}")
        logger.info(f"✓ Number of batches: {len(train_loader)}")
        
        # Try to load validation set
        try:
            val_dataset, val_loader = load_dataset(
                data_root=config.data_root,
                split='val',
                batch_size=config.batch_size
            )
            logger.info(f"✓ Validation dataset size: {len(val_dataset)}")
            has_val = True
        except:
            logger.warning("No validation set found, skipping validation")
            has_val = False
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
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
    logger.info("\n" + "="*60)
    logger.info("Starting training loop...")
    logger.info("="*60)
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{config.num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        avg_loss, loss_dict = train_one_epoch(
            model, train_loader, optimizer, device, epoch, logger
        )
        
        # Log results
        logger.info(f"\nEpoch {epoch} Training Results:")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  ITC Loss: {loss_dict['itc']:.4f}")
        logger.info(f"  ICC Loss: {loss_dict['icc']:.4f}")
        logger.info(f"  TCC Loss: {loss_dict['tcc']:.4f}")
        logger.info(f"  ITM Loss: {loss_dict['itm']:.4f}")
        logger.info(f"  LM Loss: {loss_dict['lm']:.4f}")
        
        # Validation
        if has_val and epoch % args.val_freq == 0:
            val_loss = validate(model, val_loader, device, logger)
            logger.info(f"  Validation Loss: {val_loss:.4f}")
        
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
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, avg_loss,
                f'checkpoints/checkpoint_epoch_{epoch}.pth'
            )
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, avg_loss,
            'checkpoints/latest.pth'
        )
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CASC model')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    parser.add_argument('--save-freq', type=int, default=5,
                      help='Save checkpoint every N epochs')
    parser.add_argument('--val-freq', type=int, default=1,
                      help='Run validation every N epochs')
    
    args = parser.parse_args()
    
    main(args)
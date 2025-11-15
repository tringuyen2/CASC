"""
Custom Dataset for Image-Text Pairs
Supports format: image.jpg + image.txt
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from transformers import BertTokenizer
import glob


class ImageTextPairDataset(Dataset):
    """
    Dataset for paired image and text files
    Format:
        images/image1.jpg + texts/image1.txt
        images/image2.jpg + texts/image2.txt
    """
    def __init__(self, data_root, split='train', image_size=384, max_length=77):
        """
        Args:
            data_root: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            image_size: Size to resize images
            max_length: Maximum length of text tokens
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        
        # Setup paths
        self.split_dir = os.path.join(data_root, split)
        
        # Image transforms
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Load image-text pairs
        self.samples = self._load_samples()
        
        print(f"✓ Loaded {len(self.samples)} samples for {split} split from {self.split_dir}")
    
    def _load_samples(self):
        """Load all image-text pairs from directory"""
        samples = []
        
        # Find all image files
        image_patterns = [
            os.path.join(self.split_dir, '*.jpg'),
            os.path.join(self.split_dir, '*.jpeg'),
            os.path.join(self.split_dir, '*.png'),
            os.path.join(self.split_dir, 'images', '*.jpg'),
            os.path.join(self.split_dir, 'images', '*.jpeg'),
            os.path.join(self.split_dir, 'images', '*.png'),
        ]
        
        image_files = []
        for pattern in image_patterns:
            image_files.extend(glob.glob(pattern))
        
        print(f"Found {len(image_files)} image files in {self.split_dir}")
        
        # For each image, find corresponding text file
        for img_path in image_files:
            # Get base name without extension
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Try to find text file
            text_paths = [
                os.path.join(os.path.dirname(img_path), f"{base_name}.txt"),
                os.path.join(self.split_dir, 'texts', f"{base_name}.txt"),
                os.path.join(self.split_dir, f"{base_name}.txt"),
            ]
            
            text_path = None
            for tp in text_paths:
                if os.path.exists(tp):
                    text_path = tp
                    break
            
            if text_path is None:
                print(f"Warning: No text file found for {img_path}")
                continue
            
            # Read text content
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if not text:
                    print(f"Warning: Empty text file {text_path}")
                    continue
                
                # Extract person ID from filename if possible
                # Format: personID_cameraID_frameID.jpg
                try:
                    person_id = int(base_name.split('_')[0])
                except:
                    # If can't parse, use hash of filename
                    person_id = hash(base_name) % 10000
                
                samples.append({
                    'image_path': img_path,
                    'text': text,
                    'person_id': person_id,
                    'filename': base_name
                })
                
            except Exception as e:
                print(f"Error reading {text_path}: {e}")
                continue
        
        if len(samples) == 0:
            print(f"ERROR: No valid image-text pairs found in {self.split_dir}")
            print(f"Please check your data structure:")
            print(f"  Option 1: {self.split_dir}/image1.jpg + {self.split_dir}/image1.txt")
            print(f"  Option 2: {self.split_dir}/images/image1.jpg + {self.split_dir}/texts/image1.txt")
            raise ValueError("No valid samples found!")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['image_path']}: {e}")
            # Create dummy image
            image = Image.new('RGB', (self.image_size, self.image_size), color=(128, 128, 128))
        
        # Transform image
        image = self.transform(image)
        
        # Tokenize text
        text = sample['text']
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        text_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create labels for language modeling (shift by 1)
        labels = text_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'images': image,
            'text_ids': text_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'person_id': sample['person_id'],
            'id': idx,
            'filename': sample['filename']
        }


def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item['images'] for item in batch])
    text_ids = torch.stack([item['text_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    person_ids = torch.tensor([item['person_id'] for item in batch])
    ids = torch.tensor([item['id'] for item in batch])
    
    return {
        'images': images,
        'text_ids': text_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'person_ids': person_ids,
        'ids': ids
    }


def create_data_structure_guide():
    """Print guide for data structure"""
    guide = """
    ============================================================
    Supported Data Structures
    ============================================================
    
    Option 1: Flat structure
    ------------------------
    data/
    ├── train/
    │   ├── person001_c1_f001.jpg
    │   ├── person001_c1_f001.txt
    │   ├── person001_c2_f002.jpg
    │   ├── person001_c2_f002.txt
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...
    
    Option 2: Separate folders
    ---------------------------
    data/
    ├── train/
    │   ├── images/
    │   │   ├── person001_c1_f001.jpg
    │   │   └── ...
    │   └── texts/
    │       ├── person001_c1_f001.txt
    │       └── ...
    ├── val/
    └── test/
    
    Text File Format (.txt)
    -----------------------
    Each .txt file contains a natural language description:
    
    Example (person001_c1_f001.txt):
    "A person wearing a red jacket and blue jeans, 
    carrying a black backpack"
    
    Filename Convention (recommended)
    ----------------------------------
    personID_cameraID_frameID.jpg
    personID_cameraID_frameID.txt
    
    Example:
    - 001_c1_f001.jpg → Person ID: 001, Camera: 1, Frame: 001
    - 001_c1_f001.txt → Same person's description
    
    ============================================================
    """
    print(guide)


# ==================== test_custom_dataset.py ====================
def test_custom_dataset():
    """Test the custom dataset"""
    print("="*60)
    print("Testing Custom Image-Text Pair Dataset")
    print("="*60)
    
    # Print guide
    create_data_structure_guide()
    
    # Try to load dataset
    try:
        dataset = ImageTextPairDataset(
            data_root='./data/CUHK_PEDES_images',
            split='train',
            image_size=384
        )
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        
        # Load first sample
        print(f"\nTesting first sample...")
        sample = dataset[0]
        
        print(f"✓ Sample loaded successfully!")
        print(f"  Image shape: {sample['images'].shape}")
        print(f"  Text IDs shape: {sample['text_ids'].shape}")
        print(f"  Person ID: {sample['person_id']}")
        print(f"  Filename: {sample['filename']}")
        
        # Decode text
        text = dataset.tokenizer.decode(
            sample['text_ids'], 
            skip_special_tokens=True
        )
        print(f"  Text: {text[:100]}...")
        
        # Test dataloader
        from torch.utils.data import DataLoader
        
        print(f"\nTesting DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        batch = next(iter(dataloader))
        print(f"✓ DataLoader works!")
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch text_ids shape: {batch['text_ids'].shape}")
        
        print("\n" + "="*60)
        print("✓ All tests passed! Dataset is ready for training.")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "="*60)
        print("Please check your data structure and try again.")
        print("="*60)
        
        return False


if __name__ == '__main__':
    test_custom_dataset()
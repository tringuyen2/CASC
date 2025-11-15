import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms
from transformers import BertTokenizer


class PersonSearchDataset(Dataset):
    """
    Dataset for Text-based Person Search
    Supports CUHK-PEDES, ICFG-PEDES, and RSTPReid formats
    """
    def __init__(self, data_root, split='train', image_size=384, max_length=77):
        """
        Args:
            data_root: Root directory of the dataset
            split: 'train', 'val', or 'test'
            image_size: Size to resize images
            max_length: Maximum length of text tokens
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.max_length = max_length
        
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
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        print(f"Loaded {len(self.annotations)} samples for {split} split")
    
    def _load_annotations(self):
        """Load dataset annotations"""
        # Check for annotation file
        ann_file = os.path.join(self.data_root, f'{self.split}.json')
        
        if os.path.exists(ann_file):
            # Load from JSON file
            with open(ann_file, 'r') as f:
                annotations = json.load(f)
            return annotations
        else:
            # Create dummy data for demonstration
            print(f"Warning: Annotation file {ann_file} not found!")
            print("Creating dummy dataset for demonstration...")
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy data for demonstration when real data is not available"""
        dummy_annotations = []
        
        # Create 100 dummy samples
        for i in range(100):
            person_id = i % 10  # 10 different persons
            
            # Dummy descriptions
            descriptions = [
                f"A person wearing a red jacket and blue jeans, person {person_id}",
                f"Individual in black shirt and gray pants, person {person_id}",
                f"Person with white top and dark trousers, person {person_id}",
                f"Someone in casual wear with a backpack, person {person_id}",
                f"Person wearing sports attire, person {person_id}"
            ]
            
            annotation = {
                'image_path': f'dummy_image_{i}.jpg',
                'caption': descriptions[i % len(descriptions)],
                'person_id': person_id,
                'id': i
            }
            dummy_annotations.append(annotation)
        
        return dummy_annotations
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        annotation = self.annotations[idx]
        
        # Load image
        image_path = os.path.join(self.data_root, 'images', annotation['image_path'])
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                # Create dummy image if file doesn't exist
                image = Image.new('RGB', (self.image_size, self.image_size), 
                                 color=(128, 128, 128))
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (self.image_size, self.image_size), 
                             color=(128, 128, 128))
        
        # Transform image
        image = self.transform(image)
        
        # Tokenize text
        caption = annotation['caption']
        encoded = self.tokenizer(
            caption,
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
            'person_id': annotation.get('person_id', 0),
            'id': annotation.get('id', idx)
        }


class CUHKPEDESDataset(PersonSearchDataset):
    """CUHK-PEDES Dataset"""
    def _load_annotations(self):
        """Load CUHK-PEDES format annotations"""
        ann_file = os.path.join(self.data_root, f'reid_raw.json')
        
        if not os.path.exists(ann_file):
            print(f"CUHK-PEDES annotation file not found: {ann_file}")
            return self._create_dummy_data()
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Parse CUHK-PEDES format
        annotations = []
        split_data = [item for item in data if item['split'] == self.split]
        
        for item in split_data:
            for caption in item['captions']:
                annotation = {
                    'image_path': item['file_path'],
                    'caption': caption,
                    'person_id': item['id'],
                    'id': len(annotations)
                }
                annotations.append(annotation)
        
        return annotations


class ICFGPEDESDataset(PersonSearchDataset):
    """ICFG-PEDES Dataset"""
    def _load_annotations(self):
        """Load ICFG-PEDES format annotations"""
        ann_file = os.path.join(self.data_root, f'{self.split}_data.json')
        
        if not os.path.exists(ann_file):
            print(f"ICFG-PEDES annotation file not found: {ann_file}")
            return self._create_dummy_data()
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        annotations = []
        for item in data:
            annotation = {
                'image_path': item['img_path'],
                'caption': item['caption'],
                'person_id': item['id'],
                'id': len(annotations)
            }
            annotations.append(annotation)
        
        return annotations


class RSTPReidDataset(PersonSearchDataset):
    """RSTPReid Dataset"""
    def _load_annotations(self):
        """Load RSTPReid format annotations"""
        ann_file = os.path.join(self.data_root, f'data_captions.json')
        
        if not os.path.exists(ann_file):
            print(f"RSTPReid annotation file not found: {ann_file}")
            return self._create_dummy_data()
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Filter by split
        annotations = []
        for person_id, person_data in data.items():
            if person_data['split'] != self.split:
                continue
            
            for img_name, captions in person_data['captions'].items():
                for caption in captions:
                    annotation = {
                        'image_path': os.path.join(person_data['img_path'], img_name),
                        'caption': caption,
                        'person_id': int(person_id),
                        'id': len(annotations)
                    }
                    annotations.append(annotation)
        
        return annotations


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


def get_dataloader(dataset_name, data_root, split, batch_size, num_workers=4):
    """
    Get dataloader for specified dataset
    
    Args:
        dataset_name: 'CUHK-PEDES', 'ICFG-PEDES', or 'RSTPReid'
        data_root: Root directory of dataset
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers for data loading
    """
    # Select dataset class
    if dataset_name == 'CUHK-PEDES':
        dataset_class = CUHKPEDESDataset
    elif dataset_name == 'ICFG-PEDES':
        dataset_class = ICFGPEDESDataset
    elif dataset_name == 'RSTPReid':
        dataset_class = RSTPReidDataset
    else:
        dataset_class = PersonSearchDataset
    
    # Create dataset
    dataset = dataset_class(data_root=data_root, split=split)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == 'train')
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataset
    print("Testing PersonSearchDataset...")
    
    dataset = PersonSearchDataset(
        data_root='./data/CUHK-PEDES',
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Image shape: {sample['images'].shape}")
    print(f"Text IDs shape: {sample['text_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Batch images shape: {batch['images'].shape}")
    print(f"Batch text_ids shape: {batch['text_ids'].shape}")
    
    print("\n✅ Dataset test completed successfully!")
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from config import config
from models.casc_model import CASCModel

class PersonSearchInference:
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = CASCModel(config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @torch.no_grad()
    def encode_images(self, image_paths):
        '''Encode a list of images'''
        image_features = []
        
        for img_path in tqdm(image_paths, desc='Encoding images'):
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            feat = self.model.extract_features(images=img_tensor)
            image_features.append(feat.cpu())
        
        return torch.cat(image_features, dim=0)
    
    @torch.no_grad()
    def encode_text(self, text_query):
        '''Encode a text query'''
        # Tokenize text
        encoded = self.model.text_encoder.tokenize([text_query])
        text_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Extract features
        feat = self.model.extract_features(
            text_ids=text_ids,
            attention_mask=attention_mask
        )
        
        return feat.cpu()
    
    def search(self, text_query, image_features, top_k=10):
        '''Search for images matching the text query'''
        # Encode query
        text_feat = self.encode_text(text_query)
        
        # Compute similarities
        text_feat = F.normalize(text_feat, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        
        similarities = torch.matmul(text_feat, image_features.t()).squeeze(0)
        
        # Get top-k results
        top_k_values, top_k_indices = torch.topk(similarities, min(top_k, len(similarities)))
        
        return top_k_indices.tolist(), top_k_values.tolist()

def main():
    # Example usage
    inference = PersonSearchInference('checkpoints/best_model.pth')
    
    # Example: Search in a gallery
    gallery_paths = [
        'path/to/image1.jpg',
        'path/to/image2.jpg',
        # ... more images
    ]
    
    print('Encoding gallery images...')
    image_features = inference.encode_images(gallery_paths)
    
    # Search with text query
    query = "A person wearing a red jacket and blue jeans"
    print(f'\\nSearching for: {query}')
    
    indices, scores = inference.search(query, image_features, top_k=5)
    
    print('\\nTop 5 results:')
    for i, (idx, score) in enumerate(zip(indices, scores)):
        print(f'{i+1}. Image: {gallery_paths[idx]}, Score: {score:.4f}')

if __name__ == '__main__':
    main()
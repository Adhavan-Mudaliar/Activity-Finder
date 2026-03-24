import torch
import open_clip
from PIL import Image
import os

class CLIPEncoder:
    def __init__(self, model_name="ViT-L-14", pretrained="openai", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the CLIP model. In offline mode, uses weights from weights/ dir.
        """
        self.device = device
        
        # In offline mode, we assume the weights are locally cached in the weights/ folder
        # or pointed to by the pretrained argument.
        # The user's instruction says: "Load from local weights/"
        
        # Check if local weights exist
        weight_path = os.path.join("weights", f"{model_name}_{pretrained}.pt")
        
        print(f"Loading CLIP model {model_name} from {weight_path if os.path.exists(weight_path) else 'pretrained cache'}...")
        
        try:
            # If weight_path exists, load from it. Otherwise, assume it's in the default open_clip cache location
            # which we should have populated during initial setup (as per instructions).
            if os.path.exists(weight_path):
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=weight_path, device=device
                )
            else:
                # Fallback to standard loading (likely to fail if not cached and offline)
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained, device=device
                )
            
            self.model.eval()
            # Frozen weights
            for param in self.model.parameters():
                param.requires_grad = False
                
            # Use FP16 if on GPU
            if device == "cuda":
                self.model = self.model.half()
            
            # Get embedding dimension
            self.embedding_dim = self.model.visual.output_dim
                
            print(f"CLIP model ({model_name}) loaded successfully on {device}. Dim: {self.embedding_dim}")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            raise

    @torch.no_grad()
    def encode_image(self, image_path):
        """
        Encodes a single image into a vector.
        """
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        if self.device == "cuda":
            image_input = image_input.half()
            
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

    @torch.no_grad()
    def encode_batch(self, image_paths):
        """
        Encodes a batch of images into vectors.
        """
        images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            images.append(self.preprocess(img))
            
        image_input = torch.stack(images).to(self.device)
        
        if self.device == "cuda":
            image_input = image_input.half()
            
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()

if __name__ == "__main__":
    # Test (requires weights)
    # encoder = CLIPEncoder()
    # features = encoder.encode_image("data/frames/sample/frame_000000.jpg")
    # print(features.shape)
    pass

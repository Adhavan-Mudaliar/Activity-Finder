import torch
import open_clip
import os
from src.models.clip_encoder import CLIPEncoder

class TextEncoder:
    def __init__(self, clip_encoder=None, model_name="ViT-L-14", pretrained="openai", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TextEncoder. If a clip_encoder is provided, it uses its model.
        """
        self.device = device
        if clip_encoder is not None:
            self.model = clip_encoder.model
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            # Load model and tokenizer independently if needed
            self.model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            if device == "cuda":
                self.model = self.model.half()

    @torch.no_grad()
    def encode_text(self, text):
        """
        Encodes a text query into a vector.
        """
        text_input = self.tokenizer([text]).to(self.device)
        
        # OpenCLIP expects tokens
        text_features = self.model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()

if __name__ == "__main__":
    # Test
    # encoder = TextEncoder()
    # features = encoder.encode_text("a dog running in the park")
    # print(features.shape)
    pass

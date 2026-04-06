import torch
import open_clip

device="mps"
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)

# test BFloat16
model.bfloat16()
try:
    text = open_clip.get_tokenizer("ViT-L-14")(["hello"]).to(device)
    model.encode_text(text)
    print("BFloat16 works!")
except Exception as e:
    print("BFloat16 error:", e)

# test Float16
model.half()
try:
    text = open_clip.get_tokenizer("ViT-L-14")(["hello"]).to(device)
    model.encode_text(text)
    print("Float16 works!")
except Exception as e:
    print("Float16 error:", e)


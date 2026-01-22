# zero_shot_inference.py
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model, preprocess = clip.load("ViT-L-14-findingemo", device=device)

# FindingEmo use 24 categories of emotions (Plutchik wheel, standard order)
emotions = [
    "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
    "ecstasy", "admiration", "terror", "amazement", "grief", "loathing", "rage", "vigilance",
    "serenity", "acceptance", "apprehension", "distraction", "pensiveness", "boredom",
    "annoyance", "interest"
]

text_prompts = [f"a complex social scene evoking {e}" for e in emotions]
text_tokens = clip.tokenize(text_prompts).to(device)

def predict_emotion(image_path):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        pred_idx = similarities.argmax().item()
        confidence = similarities[0, pred_idx].item()
    
    return emotions[pred_idx], confidence

# test
if __name__ == "__main__":
    test_image = "test_scene.jpg"  # Replace with a real test image path
    emotion, conf = predict_emotion(test_image)
    print(f"Predicted emotion: {emotion} (confidence: {conf:.3f})")
from PIL import Image

import torch
import torchvision.transforms as transforms

from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

import eva_vit_model
from eva_vit_model import CLIP
from open_clip.tokenizer import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'eva_base_p16'
    ckpt_path = 'path/to/ckpt'

    print(f"creating model: {model_name}")
    model = CLIP(vision_model=model_name)
    model.to(device)

    print(f"loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    print(f"reading image images/pipeline.png")
    image = Image.open("images/pipeline.png")
    image_size = 336 if '336' in model_name else 224
    preprocess = transforms.Compose([
        transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        _convert_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])
    image = preprocess(image).unsqueeze(0).to(device)

    model.eval()

    class_names = ["a diagram", "a dog", "a cat"]
    text = tokenize(class_names).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)
    print("Class name:", class_names[text_probs.squeeze().argmax()])


if __name__ == '__main__':
    main()

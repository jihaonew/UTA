import os
from tqdm import tqdm
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from timm.data.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

import eva_vit_model
from eva_vit_model import CLIP
from open_clip.tokenizer import tokenize
from imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True

    print(f"creating model: {args.model}")
    model = CLIP(vision_model=args.model)
    model.to(device)

    print(f"loading checkpoint from {args.ckpt_path}")
    state_dict = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=True)

    def _convert_image_to_rgb(image):
        return image.convert("RGB")

    val_transform = transforms.Compose([
        transforms.Resize(args.image_size, transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD)
    ])

    val_dataset = datasets.ImageFolder(os.path.join(args.imagenet_path, 'val'), transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers)

    model.eval()
    classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, device)
    top1, top5 = zero_shot_eval(model, classifier, val_loader, device)
    print(f'ImageNet zeroshot top1: {top1:.4f}, top5: {top5:.4f}')


def zero_shot_classifier(model, classnames, templates, device):
    tokenizer = tokenize
    
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenizer(texts).to(device=device)  # tokenize
            with torch.cuda.amp.autocast():
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def zero_shot_eval(model, classifier, dataloader, device):
    top1, top5, n = 0., 0., 0.
    with torch.no_grad():
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device)
            target = target.to(device=device)

            with torch.cuda.amp.autocast():
                image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet zero shot evaluations', add_help=False)
    parser.add_argument('--imagenet-path', default='path/to/imagenet', type=str, help='path to imagenet dataset')
    parser.add_argument('--ckpt-path', default='path/to/ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--model', default='eva_base_p16', type=str, help='model')
    parser.add_argument('--image-size', default=224, type=int, help='image size for evaluation')
    parser.add_argument('--workers', default=8, type=int)
    args = parser.parse_args()
    main(args)

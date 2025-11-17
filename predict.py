import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from hyper_new import HyperVisionNet 

def get_inference_transforms(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

def generate_heatmap(original_image, cam_features, class_idx, model_head):
    """Generates and overlays a heatmap on the original image using GNN-CAM."""

    cam_weights = model_head[-1].weight[class_idx].detach().cpu()

    cam = cam_features.cpu() @ cam_weights

    num_patches = cam.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError("Cannot form a square grid from patch features. Check model architecture.")
        
    cam = cam.reshape(grid_size, grid_size)

    cam = cam.numpy()
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-6)
    cam = np.uint8(cam * 255)

    heatmap_target_size = original_image.size 
    heatmap = cv2.resize(cam, heatmap_target_size)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    original_img_np = np.array(original_image).astype(np.float32)
    heatmap_np = heatmap.astype(np.float32)

    superimposed_img = (original_img_np * 0.5 + heatmap_np * 0.5).astype(np.uint8)
    
    return superimposed_img


def main(args):    
    device_name = args.device
    
    if device_name == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        
    print(f"--- Using device: {device} ---")


    num_classes = 4
    model = HyperVisionNet(
        num_classes=num_classes, 
        patch_embed_dim=args.patch_dim,
        gnn_hidden=args.hidden,
        gnn_layers=args.layers,
        k=args.k,
        num_clusters=args.num_clusters,
        use_hyperedges=True 
    ).to(device)

    model.load_state_dict(torch.load(args.model_weights, map_location=device))
    model.eval()

    img_transform = get_inference_transforms(args.img_size)
    original_image = Image.open(args.image).convert("RGB")

    image_tensor = img_transform(original_image).unsqueeze(0).to(device)

    print("--- Running model for prediction and visualization ---")

    with torch.no_grad():
        logits, cam_features = model(image_tensor, return_cam_features=True)
    
    probs = torch.softmax(logits, dim=1)
    pred_prob, pred_class_idx = torch.max(probs, dim=1)
    

    class_names = ["giloma" , "meningioma", "notumor", "pituitary"] 
    pred_class_name = class_names[pred_class_idx.item()]
    
    print(f"Prediction: {pred_class_name} (Confidence: {pred_prob.item():.4f})")
    heatmap_image = generate_heatmap(
        original_image, 
        cam_features, 
        pred_class_idx.item(), 
        model.head
    )
    
    output_filename = "heatmap_output.png"
    Image.fromarray(heatmap_image).save(output_filename)
    
    print(f"--- Saved heatmap to {output_filename} ---")
    
    plt.imshow(heatmap_image)
    plt.title(f"Prediction: {pred_class_name}")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and Visualization for HyperVisionNet")
    
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to the trained .pt file.")
    
    parser.add_argument("--img_size", type=int, default=224, help="Image size used during training.")
    parser.add_argument("--patch_dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--k", type=int, default=12) 
    parser.add_argument("--num_clusters", type=int, default=8)
    parser.add_argument("--device", type=str, default="mps", 
                        choices=["mps", "cuda", "cpu"], help="Device to use for inference.")
    
    args = parser.parse_args()

    import os
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    
    main(args)
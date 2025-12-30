import os
import json
import torch
import time
from src.models import NeuralStyleTransfer
from src.nst_utils import ImageHandler
from src.criterion import Criterion
from src.data_validation import TrainRequest
from torch import optim
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# HYPERPARAMETERS TO SET
# model option ['vgg19', 'vgg16' 'resnet50', 'resnet101' inception_v3]
model_name = "resnet50"
pretrained_weights_path = None
content_weight = 1
style_weight = 1e10
pooling = 'ori'  # options ['avg', 'max', 'ori']

# # for vgg
# content_layers = ["2"]
# style_layers = ["8"]

# # for inception
# content_layers = ["4"]   
# style_layers = ["8"]

# for resnet
# [[x][y]] get bottleneck in x and layer in y
content_layers = [[2][-1]]
style_layers = [[2][-1]]

# instance
nst = NeuralStyleTransfer(model_name, pretrained_weights_path=pretrained_weights_path, pooling=pooling, device=device).to(device)
print(nst)
criterion = Criterion(content_weight=content_weight, style_weight=style_weight)
image_handler = ImageHandler()

# get metadata for generated images
def load_existing_metadata(metadata_filename):
    if os.path.exists(metadata_filename):
        with open(metadata_filename, 'r') as json_file:
            metadata = json.load(json_file)
            if "sessions" not in metadata:
                metadata["sessions"] = [] 
            return metadata
    return {"sessions": []} 

# train nst
def train(request: TrainRequest):
    # load content and style images
    content_image = image_handler.load_image(request.content_image_path, image_handler.transform).to(device)
    style_image = image_handler.load_image(request.style_image_path, image_handler.transform).to(device)

    # content_image = prep(content_image).to(device)
    # style_image = prep(style_image).to(device)

    # prepare output image
    output = content_image.clone().to(device)
    output.requires_grad = True
    optimizer = optim.AdamW([output], lr=0.05)
    
    # extract features
    content_features = nst(content_image.to(device), layers=content_layers)
    style_features = nst(style_image.to(device), layers=style_layers)

    max_epochs = 15000
    print(f'--------------------- Start Training ---------------------')
    generated_image_name = ""
    
    # create output directory specific to the model
    model_output_dir = f"outputs/generate_{model_name}"
    os.makedirs(model_output_dir, exist_ok=True)

    # load existing metadata
    metadata_filename = "outputs/new_metadata.json"
    metadata = load_existing_metadata(metadata_filename)

    # prepare new session metadata
    new_session = {
        "model_name": model_name,
        "is_finetuned": pretrained_weights_path is not None,
        "pooling": pooling if pooling is not None else "avg",
        "content_image": request.content_image_path,
        "style_image": request.style_image_path,
        "content_weight": content_weight,
        "style_weight": style_weight,
        "content_layers": content_layers,
        "style_layers": style_layers,
        "generated_images": [],
        "loss_values": [],
        "ssim_values": []
    }

    # Record start time
    start_time = time.time()
    
    # Initialize loss tracking list
    losses = []

    for epoch in range(1, max_epochs + 1):
        # output_features = nst(output, layers=["4", "8"])
        output_contents = nst(output, layers=content_layers)
        output_styles = nst(output, layers=style_layers)
        # Calculate loss (returns tuple: total_loss, content_loss, style_loss)
        total_loss, content_loss, style_loss = criterion.criterion(
            content_features, 
            style_features, 
            output_contents,
            output_styles
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track losses
        if epoch % 100 == 0 or epoch == 1:
            total_loss_value = total_loss.item()
            content_loss_value = content_loss.item()
            style_loss_value = style_loss.item()
            losses.append(total_loss_value)
            print(f"Epoch {epoch}/{max_epochs}, Loss: {total_loss_value:.4f} (Content: {content_loss_value:.4f}, Style: {style_loss_value:.4f})")
        ssim_value = image_handler.calculate_ssim(output, content_image)
        print(f"Epoch: {epoch:5} | SSIM: {ssim_value:.5f}\n")
        
        # save output images at specific epochs
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if epoch % 500 == 0:
            output_image_path = f"{model_output_dir}/output_epoch_{epoch}_{timestamp}.png"
            image_handler.save_image(output, output_image_path)
            generated_image_name = f"output_epoch_{epoch}.png"

            # add generated image metadata
            new_session["generated_images"].append({
                "epoch": epoch,
                "filename": output_image_path,
                "timestamp": timestamp
            })
            # Store loss and SSIM for this epoch
            new_session["loss_values"].append(total_loss_value)
            new_session["ssim_values"].append(ssim_value)

    # Calculate training time and add it to metadata
    end_time = time.time()
    training_time = end_time - start_time
    new_session["training_time"] = training_time  # in seconds

    # append new session to existing metadata
    metadata["sessions"].append(new_session)

    # save updated metadata back to the JSON file
    with open(metadata_filename, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    return {"message": "Training completed!", "generated_image_name": model_output_dir+generated_image_name, "training_time": training_time}

def upload_and_train(content_image_path: str, style_image_path: str):
    # Create TrainRequest object and initiate training
    train_request = TrainRequest(
        content_image_path=content_image_path,
        style_image_path=style_image_path
    )
    result = train(train_request)
    print(result)

def main(content_image_path: str, style_image_path: str):
    upload_and_train(content_image_path, style_image_path)

# single generate
if __name__ == "__main__":
    content_image_path = "outputs/hasil/44/content.jpg" 
    style_image_path = "outputs/hasil/44/style.jpg" 
    
    main(content_image_path, style_image_path)

# # batch generate
# if __name__ == "__main__":
#     content_folder = "data/_content"
#     style_folder = "data/_style"

#     content_image_paths = [os.path.join(content_folder, f) for f in os.listdir(content_folder) if f.endswith((".jpg", ".png"))]
#     style_image_paths = [os.path.join(style_folder, f) for f in os.listdir(style_folder) if f.endswith((".jpg", ".png"))]

#     os.makedirs("outputs", exist_ok=True)

#     for content_image_path in content_image_paths:
#         for style_image_path in style_image_paths:
#             print(f"Processing Content: {content_image_path}, Style: {style_image_path}")
#             try:
#                 main(content_image_path, style_image_path)
#             except Exception as e:
#                 print(f"Error processing {content_image_path} with {style_image_path}: {e}")
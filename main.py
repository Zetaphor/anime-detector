import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os

def load_image(image_path):
    return Image.open(image_path)

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model.to('cpu'), processor  # Explicitly move model to CPU

def detect_image_type(image_path, model, processor):
    image = load_image(image_path)

    # Expanded text input to include more categories
    text = [
        "an anime character",
        "a cartoon animal",
        "a photograph of a person",
    ]
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

    # Ensure inputs are on CPU
    inputs = {k: v.to('cpu') for k, v in inputs.items()}

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)

    # Get the probabilities for each category
    category_probs = [prob.item() for prob in probs[0]]

    # Calculate relative probabilities
    total_prob = sum(category_probs)
    relative_probs = [prob / total_prob for prob in category_probs]

    return dict(zip(text, relative_probs))

def interpret_result(image_name, probs):
    print(f"\nImage: {image_name}")
    print("Relative probabilities:")
    for category, prob in probs.items():
        print(f"  {category}: {prob:.2%}")

    max_category = max(probs, key=probs.get)
    max_prob = probs[max_category]

    if max_prob > 0.4:
        print(f"The image is most likely {max_category}.")
    else:
        print("The image type is uncertain.")

def main():
    model, processor = load_clip_model()

    images_dir = "images"
    if not os.path.exists(images_dir):
        print(f"Error: The '{images_dir}' directory does not exist.")
        return

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print(f"No image files found in the '{images_dir}' directory.")
        return

    print(f"Processing {len(image_files)} images...")

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        try:
            probs = detect_image_type(image_path, model, processor)
            interpret_result(image_file, probs)
        except Exception as e:
            print(f"\nError processing {image_file}: {str(e)}")

    print("\nAll images processed.")

if __name__ == "__main__":
    main()

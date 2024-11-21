import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_captions_for_folder(input_folder, output_folder):
    # Initialize the pre-trained processor and model
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            text_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

            # Open and preprocess the image
            raw_image = Image.open(image_path).convert('RGB')

            # Unconditional image captioning
            inputs = processor(raw_image, return_tensors="pt")
            out = model.generate(**inputs)

            # Decode and save the caption
            caption = processor.decode(out[0], skip_special_tokens=True)
            with open(text_path, 'w') as f:
                f.write(caption)
                print(f"Generated caption for {filename}: {caption}")

if __name__ == "__main__":
    input_folder = R'E:\lh\data\fiveK\input\JPG\480p'  # Replace with your folder path
    output_folder = 'blipcaplarge'  # Replace with your desired output folder path
    generate_captions_for_folder(input_folder, output_folder)
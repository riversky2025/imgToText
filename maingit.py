import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

def generate_captions_for_folder(input_folder, output_folder):
    # Initialize the pre-trained processor and model
    processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            text_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".txt")

            # Open and preprocess the image
            image  = Image.open(image_path).convert('RGB')

            # Unconditional image captioning
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(pixel_values=pixel_values, max_length=50)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # Decode and save the caption
            with open(text_path, 'w') as f:
                f.write(caption)
                print(f"Generated caption for {filename}: {caption}")

if __name__ == "__main__":
    input_folder = R'E:\lh\data\fiveK\input\JPG\480p'  # Replace with your folder path
    output_folder = 'gitlagecap'  # Replace with your desired output folder path
    generate_captions_for_folder(input_folder, output_folder)
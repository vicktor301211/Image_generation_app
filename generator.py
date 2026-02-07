import torch
from diffusers import StableDiffusionPipeline

def dummy_safety_checker(images, clip_input):
    return images, [False] * len(images)


def generate_image(
    prompt="A beautiful tree in a sunny forest",
    output_path="tree.png"
):
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)

    pipe.safety_checker = dummy_safety_checker

    image = pipe(prompt).images[0]
    image.save(output_path)

    print(f"Изображение сохранено как {output_path}")

if __name__ == "__main__":
    generate_image()
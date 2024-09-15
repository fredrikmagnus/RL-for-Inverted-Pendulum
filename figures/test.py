from PIL import Image, ImageChops, ImageSequence
def crop_gif(input_path, output_path):
    # Read the GIF
    gif = Image.open(input_path)
    frames = []

    # Iterate through the frames
    for frame in ImageSequence.Iterator(gif):
        # Crop the frame to remove whitespace
        bbox = ImageChops.invert(frame.convert("RGB")).getbbox()
        cropped_frame = frame.crop(bbox)
        frames.append(cropped_frame)

    # Save the frames back to a new GIF
    frames[0].save(output_path, save_all=True, append_images=frames[1:], loop=gif.info['loop'], duration=gif.info['duration'])

# Example usage
crop_gif('figures/animation_up.gif', 'figures/output.gif')
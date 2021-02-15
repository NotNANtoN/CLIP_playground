import argparse
import time
import os
import subprocess
import json

import torch

#from deep_daze_repo.deep_daze.deep_daze import Imagine
from deep_daze import Imagine


def create_text_path(text=None, img=None, encoding=None):
    if text is not None:
        input_name = text.replace(" ", "_")
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


def run(text=None, img=None, encoding=None, name=None, image_width=256, **args):
    input_name = ""
    if name is not None:
        input_name += name
    if text is not None:
        input_name += text.replace(" ", "_")
    if img is not None:
        if isinstance(img, str):
            input_name += "_" + "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name += "_PIL_img"
    if input_name == "":
        input_name += "your_encoding"
    
    
    original_dir = os.getcwd()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    name = os.path.join("deepdaze", str(image_width), time_str + "_" + input_name)
    os.makedirs(name, exist_ok=True)
    # copy start image to folder
    args = dict(args)
    if "start_image_path" in args:
        subprocess.run(["cp", args["start_image_path"], name])
    if img is not None and isinstance(img, str):
        subprocess.run(["cp", img, name])
    os.chdir(name)
    # save hyperparams:
    with open("hyperparams.json", "w+") as f:
        json.dump(args, f)

    try:
        imagine = Imagine(
            text=text,
            epochs = 12,
            image_width=image_width,
            save_progress=True,
            open_folder=True,
            start_image_train_iters=200,
            **args
           )
        # set goal
        #if encoding is None and text is not None and img is not None:
            # merge img and text
        #    encoding = (imagine.create_img_encoding(img) + imagine.create_text_encoding(text)) / 2
        
        #imagine.set_clip_encoding(text=text, img=img, encoding=encoding)
        
        # train
        imagine()
        # make mp4
        subprocess.run(["ffmpeg", "-i", input_name + ".000%03d.png", "-pix_fmt", "yuv420p", input_name + ".mp4"])
        # save
        torch.save(imagine.cpu(), "model.pt")
        del imagine

    finally:
        os.chdir(original_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--num_layers", default=44, type=int)
parser.add_argument("--image_width", default=256, type=int)
parser.add_argument("--gradient_accumulate_every", default=2, type=int)
parser.add_argument("--save_every", default=25, type=int)

args = parser.parse_args()
args = vars(args)


run(text="Instagram", **args)
run(text="Instagram addiction", **args)

run("The scream by Edward Munch")

run(text="A climber climbing a large mountain", **args)
run(text="Surfing a big wave in the ocean", **args)
run(text="A peaceful walk in the forest", **args)
run(text="Torment in hell by demonic creatures", **args)

run(text="A monkey painting a painting", **args)
run(text="Chaos", **args)

run(text="The universe", **args)
run(text="The milky way", **args)
run(text="A photo of a supernova", **args)
run(text="Wondering about life while looking at the night sky", **args)

run(text="An image of a dog having a spiritual experience", **args)

quit()
#run(text="Love is the answer!", img="hot-dog.jpg")

run(text="Magic Mushrooms", **args)
run(text="Mescaline", **args)
run(text="Salvia Divinorum", **args)
run(text="DMT", **args)


run(text="A psychedelic experience on magic mushrooms", **args)
run(text="A psychedelic experience on LSD", **args)
run(text="A psychedelic experience on Mescaline", **args)
run(text="A psychedelic experience on Salvia Divinorum", **args)


run(text="LSD", **args)
run(text="Psychedelics", **args)
run(text="Trip", **args)
run(text="Psychedelic Trip", **args)


quit()

#run("The logo of a company named AdaLab", start_image_path="logo.png")
#run("The logo of a company named AdaLab", start_image_path="Logo_full.png")
#run("A photo of the logo of the A.I. startup AdaLab")
#run("The logo of a company named AdaLab")

#run("A photo of the logo of the company AdaLab")
#run("The people that work at the A.I. startup AdaLab")

#run("A photo of the logo of the company AdaLab")
#run("A photo of the logo of the AI company AdaLab")
#run("A photo of the logo of the startup AdaLab")
run("Adalab")

run("Anton")
run("Angelie")
run("Pia")
run("Sebastian")
run("Florian")

run("Nadia")
run("Lilli")
run("Ouzo")

run("Emilia")
run("Martin")
run("Heidrun")
run("A photo of Martin, Heidrun, Anton, Emilia, Johanna, Fiona, and Luca - a happy family.")

run("Artificial Intelligence")

#run("A photo of lantern bread")
#run("A photo of a strawberry hamburger")
run("A photo of weird food")
run("A photo of tasty food")

#run("A photo of my best friend")
#run("A photo of my worst enemy")
#run("A photo of me")
run("God")
run("Satan")
run("Demon")
run("A photo of God")
run("A photo of Satan")

run("Bliss")
run("Hatred")
run("Love")
run("Vengeance")


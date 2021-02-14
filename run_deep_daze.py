import time
import os
import subprocess

import torch

from deep_daze_repo.deep_daze.deep_daze import Imagine


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


def run(text=None, img=None, encoding=None, name=None, **args):
    input_name = ""
    if name is not None:
        input_name += name
    if text is not None:
        input_name += text.replace(" ", "_")
    if img is not None:
        if isinstance(img, str):
            input_name += "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name += "_PIL_img"
    else:
        input_name += "your_encoding"
    
    
    original_dir = os.getcwd()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    name = time_str + "_" + input_name
    os.makedirs(name, exist_ok=True)
    # coppy start image to folder
    args = dict(args)
    if "start_image_path" in args:
        subprocess.run(["cp", args["start_image_path"], name])
    if img is not None and isinstance(img, str):
        subprocess.run(["cp", img, name])
    os.chdir(name)

    try:
        imagine = Imagine(            
            num_layers = 44,
            batch_size = 8,
            gradient_accumulate_every = 2,
            epochs = 12,
            image_width=256,
            save_progress=True,
            open_folder=True,
            start_image_train_iters=200,
            save_every=50,
            **args
           )
        # set goal
        if encoding is None and text is not None and img is not None:
            # merge img and text
            encoding = (imagine.create_img_encoding(img) + imagine.create_text_encoding(text)) / 2
        
        imagine.set_clip_encoding(text=text, img=img, encoding=encoding)
        
        # train
        imagine()
        # save
        torch.save(imagine.cpu(), "model.pt")
        del imagine

    finally:
        os.chdir(original_dir)

        
#run(text="Love is the answer!", img="hot-dog.jpg")

run(text="Magic Mushrooms")
run(text="Ketamine")
run(text="Mescaline")
run(text="Salvia Divinorum")
run(text="DMT")
run(text="2-CB")
run(text="MDMA")
run(text="Ecstacy")


run(text="A psychedelic experience on magic mushrooms")
run(text="A psychedelic experience on LSD")
run(text="A psychedelic experience on Mescaline")
run(text="A psychedelic experience on Salvia Divinorum")


run(text="LSD")
run(text="Psychedelics")
run(text="Trip")
run(text="Psychedelic Trip")


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


import argparse
import time
import os
import subprocess
import json
import sys

import torch

#from deep_daze_repo.deep_daze.deep_daze import Imagine
#from deep_daze import Imagine
sys.path.append("../deepdaze/")
from deep_daze_repo.deep_daze.deep_daze import Imagine


def create_text_path(text=None, img=None, encoding=None):
    if text is not None:
        input_name = text.replace(" ", "_")[:77]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1])
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name


def run(text=None, img=None, encoding=None, name=None, image_width=256, **args):
    input_name = create_text_path(text=text, img=img, encoding=encoding)
    
    # switch to own folder
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
        if args["create_story"]:
            args["iterations"] = 2100
        imagine = Imagine(
            text=text,
            image_width=image_width,
            save_progress=True,
            start_image_train_iters=200,
            open_folder=False,
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
        subprocess.run(["ffmpeg", "-i", '"' + input_name + ".000%03d.png" + '"', "-pix_fmt", "yuv420p", input_name + ".mp4"])
        # save
        torch.save(imagine.cpu(), "model.pt")
        del imagine

    finally:
        os.chdir(original_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--num_layers", default=44, type=int)
parser.add_argument("--image_width", default=256, type=int)
parser.add_argument("--gradient_accumulate_every", default=1, type=int)
parser.add_argument("--save_every", default=20, type=int)
parser.add_argument("--epochs", default=10, type=int)

# for 512: 
    # bs==1,  num_layers==24 - CRASH
    # bs==1,  num_layers==22 - 7.96 GB
    # bs==2,  num_layers==20 - 7.5 GB
    # bs==16, num_layers==16 - 6.5 GB
    # bs==32, num_layers==16 - CRASH

# default grad_acc==3
# for 256:
    # bs==8, num_layers==48 - 5.3 GB
    # bs==16, num_layers==48 - 5.46 GB - 2.0 it/s
    # bs==32, num_layers==48 - 5.92 GB - 1.67 it/s
    # bs==8, num_layers==44 - 5 GB - 2.39 it/s
    # bs==32, num_layers==44, grad_acc==1 - 5.62 GB - 4.83 it/s
    # bs==96, num_layers==44, grad_acc==1 - 7.51 GB - 2.77 it/s
    # bs==32, num_layers==66, grad_acc==1 - 7.09 GB - 3.7 it/s

args = parser.parse_args()
args = vars(args)


def run_from_file(path, **args):
    with open(path, 'r') as f:
        data = f.read()
    texts = data.split("\n")
    # filter empty
    texts = [text for text in texts if len(text) > 0]
    # filter comments
    texts = [text for text in texts if text[0] != "#"]
    
    for text in texts:
        run(text=text, **args)
    



run_from_file("dreams_male_college.txt", create_story= True, **args)

run_from_file("dreams_female_college.txt", create_story=True, **args)



quit()
#run(text="A neural network.", **args)
#run(text="An artificial neural network generating images.", **args)
#run(text="A LinkedIn post.", **args)
#run(text="An Instagram post", **args)
#run(text="The website of the A.I. startup AdaLab.", **args)
#run(text="Instagram addiction", **args)

#run(text="Florian", **args)
#run(text="Pia", **args)
#run(text="Alex", **args)
#run(text="Alexander Busch", **args)
#run(text="Friedi", **args)

#run(text="Consciousness", **args)
#run(text="Enlightenment", **args)
#run(text="Depression", **args)
#run(text="Multiple personality disorder", **args)
#run(text="Schizophrenia", **args)
#run(text="Hearing voices", **args)
#run(text="A schizoprhenic episode", **args)
#run(text="Mania", **args)

#run(text="A crying man.", **args)
#run(text="A crying woman.", **args)
#run(text="A terrorist.", **args)
run(text="A photo of a terrorist.", **args)
run(text="A criminal.", **args)
run(text="A photo of a criminal.", **args)
run(text="A cute person.", **args)
run(text="A poor person.", **args)
run(text="A rich person.", **args)
run(text="A beautiful person.", **args)
run(text="An ugly person.", **args)

run(text="Entering the gates of Heaven.", **args)
run(text="Meeting God", **args)
run(text="The final judgement.", **args)
run(text="The all-seeing eye.", **args)
run(text="The all-seeing tree.", **args)
run(text="The blue all-seeing tree. A blue tree with a large eye in its stem", **args)
run(text="Being born", **args)
run(text="The process of dying", **args)

run(text="Anton Wiehe", **args)
run(text="Elon Musk", **args)
run(text="Bill Gates", **args)

run(text="Rock climbing", **args)
run(text="Bouldering", **args)

run(text="Happiness", **args)

quit()

#run(text="Instagram", **args)
run(text="Someone, addicted to Instagram, is scrolling on their phone.", **args)

run(text="The scream by Edvard Munch", **args)

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
run(text="A dancing robot", **args)

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


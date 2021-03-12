import argparse
import os
import time
import subprocess
import copy
import sys

sys.path.append("../")
from big_sleep_repo.big_sleep.big_sleep import Imagine
#from big_sleep import Imagine

def underscorify(text):
    no_punctuation = text.replace(".", "")
    spaces_to_underline = no_punctuation.replace(" ", "_")
    no_lead_trailing_underscores = spaces_to_underline.strip('-_')
    no_commas = no_lead_trailing_underscores.replace(",", "")
    return no_commas



def run(self, text, args, img=None, **kwargs):
    if args is None:
        args = {}
    args = copy.copy(args)
    for key in kwargs:
        args[key] = kwargs[key]

    orig_os = os.getcwd()
    name = underscorify(name)

    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    folder_name = time_str + name
    folder_name = os.path.join("bigsleep", str(args["image_size"]), folder_name)
    os.makedirs(folder_name, exist_ok=True)
    # copy image for feature extraction to folder
    if img is not None and isinstance(img, str):
        img_new_name = img.split("/")[-1] # only take end path
        remove_list = [")", "(", "[", "]", '"', "'"]
        for char in remove_list:
            img_new_name = img_new_name.replace(char, "")
        subprocess.run(["cp", img, os.path.join(folder_name, img_new_name)])
        img = img_new_name
    os.chdir(folder_name)

    dream = Imagine(
            text=text,
            img=img,
            save_progress=True,
            save_best=True,
            save_every = 25,
            **args
           )
    dream()

    file_names = '"' + name + ".%d.png" + '"'
    movie_name = '"' + name + ".mp4" + '"'
    subprocess.run(" ".join(["ffmpeg", "-i", file_names, "-pix_fmt", "yuv420p", movie_name]), shell=True)

    os.chdir(orig_os)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--image_size", type=int, default=512, choices=[128, 256, 512])
parser.add_argument("--gradient_accumulate_every", type=int, default=1)
parser.add_argument("--epochs", type=int, default=5)


args = parser.parse_args()
args = vars(args)


args["image_size"] = 256

# some prompts
lama = "A llama wearing a scarf and glasses, reading a book in a cozy cafe."
wizard = "A wizard in blue robes is painting a completely red image in a castle."
consciousness = "Consciousness"
bathtub_love = "I love you like a bathtub full of ice cream."
bathtub = "A bathtub full of ice cream."


prompt = lama

# try some stuff
run(None, img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg")
run("WARNING: This is just a test!", args=args, epochs=1)


args["num_cutous"] = 4
run(wizard, args=args)
run(consciousness, args=args)
args["num_cutous"] = 8
run(consciousness, args=args)
run(wizard, args=args)
args["num_cutous"] = 16
run(consciousness, args=args)
run(wizard, args=args)
args["num_cutous"] = 32
run(consciousness, args=args)
run(wizard, args=args)
args["num_cutous"] = 64
run(consciousness, args=args)
run(wizard, args=args)
args["num_cutous"] = 128



run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("A painting of a sunrise.", args=args)
run("The logo of an A.I. startup named AdaLab", args=args)


args["max_classes"] = 15
args["class_temperature"] = 0.1
run(wizard, args=args)
run(consciousness, args=args)
args["class_temperature"] = 0.5
run(consciousness, args=args)
run(wizard, args=args)
args["class_temperature"] = 2.0
run(consciousness, args=args)
run(wizard, args=args)
args["class_temperature"] = 4.0
run(consciousness, args=args)
run(wizard, args=args)
args["class_temperature"] = 10.0
run(consciousness, args=args)
run(wizard, args=args)
args["class_temperature"] = 2.0


args["max_classes"] = 1
run(wizard, args=args)
run(consciousness, args=args)
args["max_classes"] = 3
run(wizard, args=args)
run(consciousness, args=args)
args["max_classes"] = 10
run(wizard, args=args)
run(consciousness, args=args)
args["max_classes"] = 30
run(wizard, args=args)
run(consciousness, args=args)
args["max_classes"] = 100
run(wizard, args=args)
run(consciousness, args=args)


args["max_classes"] = 15
run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("A painting of a sunrise.", args=args)
run("The logo of an A.I. startup named AdaLab", args=args)
args["max_classes"] = None


args["bilinear"] = True
run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("A painting of a sunrise.", args=args)
run("The logo of an A.I. startup named AdaLab", args=args)
args["bilinear"] = False


args["experimental_resample"] = True
run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("A painting of a sunrise.", args=args)
run("The logo of an A.I. startup named AdaLab", args=args)
args["experimental_resample"] = False


args["ema_decay"] = 0.0
run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("The logo of an A.I. startup named AdaLab")
run("A painting of a sunrise.", args=args)
args["ema_decay"] = 0.5


args["gradient_accumulate_every"] = 8
run("A psychedelic experience on LSD", args=args)
run(wizard, args=args)
run(lama, args=args)
run(consciousness, args=args)
run("The sun setting spectaculously over the beautiful ocean.", args=args)
run("A painting of a sunset.", args=args)
run("The logo of an A.I. startup named AdaLab")
run("A painting of a sunrise.", args=args)


quit()

run("Consciousness")
run("Enlightenment")
run("Depression")
run("Multiple personality disorder")
run("Schizophrenia")

run("The all-seeing eye.")
run("The all-seeing tree.")
run("The blue all-seeing tree. A blue tree with a large eye in its stem")
run("Being born")
run("The process of dying")

run("A psychedelic experience on magic mushrooms")
run("A psychedelic experience on LSD")
run("A psychedelic experience on Mescaline")
run("A psychedelic experience on Salvia Divinorum")

quit()

if args.mode == 0:
    run("A photo of the logo of an A.I. startup named AdaLab")
    run("The logo of an A.I. startup named AdaLab")
    run("A photo of Adalab")
    run("A photo of the people working at AdaLab")
    run("A photo of the logo of Adalab, a successfull A.I. startup")
    run("Adalab")
    
    run("God")
    run("Satan")
    
    run("A photo of Anton")
    run("A photo of Florian")
    run("A photo of Sebastian")
    run("A photo of Pia")
    run("A photo of Angelie")
    
    run("Anton")
    run("Florian")
    run("Sebastian")
    run("Pia")
    run("Angelie")
    
    run("A photo of a woman named Anica")
    run("A photo of a woman named Nadia")
    run("A photo of a woman named Lilli")
    run("A photo of a woman named Friedi")

    run("Artificial Intelligence")
    run("Robots")

    run("A monkey painting a painting")
    run("The seven deathly sins")
    run("The seven sins")

    run("Love")
    run("Freedom")
    run("Bliss")
    run("Serenity")
    run("Enlightenment")
    run("Buddha")
    
    run("LSD")
    run("Magic Mushrooms")
    run("Ketamine")
    run("Mescaline")
else:
    run("A photo of lantern bread")
    run("A photo of a strawberry hamburger")
    run("A photo of weird food")
    run("A photo of tasty food")
    
    run("A photo of my mum")
    run("A photo of me")
    run("A photo of me, myself, and I")
    run("A photo of God")
    run("A photo of Satan")
    
    run("A photo of a man named Anton")
    run("A photo of a man named Florian")
    run("A photo of a man named Sebastian")
    run("A photo of a woman named Pia")
    run("A photo of a woman named Angelie")
    
    run("A photo of a dog named Ouzo")
    run("A photo of Ouzo, a type of dog")
    run("A photo of a cat named Leo")
    run("A photo of Leo, a type of cat")
    
    run("A photo of a cat")
    run("A photo of a dog")
    run("A photo of mountains")
    
    run("A scary old witch")
    run("A demon")
    run("A demonic screech")
    
    run("Uncertainty")
    run("Depression")
    run("Anxiety")
    run("Insecurity")
    run("Nightmare")

    run("Loki")
    run("Zeus")
    
    run("Psychedelics")
    run("Psychedelia")
    run("A psychedelic trip")
    run("A psychedelic experience")
    run("Me, eating some fruit while high on LSD")
    run("Me, eating some fruit while high on mushrooms")
    run("High on mushrooms")

import argparse
import os
import time
import subprocess

from big_sleep import Imagine


class Runner:
    def __init__(self, args):
        self.args = args
        self.size = args.size
        
    def __call__(self, text, **args):
        orig_os = os.getcwd()
        name = text.replace(" ", "_")

        folder_name = name + time.strftime("%X", time.gmtime())
        folder_name = os.path.join("bigsleep", str(self.size), folder_name)
        os.makedirs(folder_name, exist_ok=True)
        os.chdir(folder_name)

        dream = Imagine(
                text = text,
                image_size = self.size,
                gradient_accumulate_every = 2,
                epochs = 10,

                save_progress=True,
                save_best=True,
                save_every = 25,
                **args
               )
        dream()
        
        name = text.replace(" ", "_")
        file_names = '"' + name + ".000%03d.png" + '"'
        movie_name = '"' + name + ".mp4" + '"'
        subprocess.run(" ".join(["ffmpeg", "-i", file_names, "-pix_fmt", "yuv420p", movie_name]), shell=True)

        os.chdir(orig_os)
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=int, default=0, choices=[0, 1])
parser.add_argument("--size", type=int, default=512, choices=[128, 256, 512])
args = parser.parse_args()

run = Runner(args)


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

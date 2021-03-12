import argparse
import time
import os
import subprocess
import json
import sys

import torch

#from eep_daze_repo.deep_daze.deep_daze import Imagine
#frgs)
from deep_daze import Imagine
sys.path.append("../deepdaze/")
from deep_daze_repo.deep_daze.deep_daze import Imagine


def create_text_path(text=None, img=None, encoding=None):
    if text is not None:
        input_name = text.replace(" ", "_")[:77]
    elif img is not None:
        if isinstance(img, str):
            input_name = "".join(img.replace(" ", "_").split(".")[:-1]) # replace spaces by underscores, remove img extension
            input_name = input_name.split("/")[-1]  # only take img name, not path
        else:
            input_name = "PIL_img"
    else:
        input_name = "your_encoding"
    return input_name

import copy


def run(text=None, img=None, encoding=None, name=None, args=None, **kwargs):
    if args is None:
        args = {}
    args = copy.copy(args)
    for key in kwargs:
        args[key] = kwargs[key]
    
    if img is not None and isinstance(img, str):
        pass
        #img = img.replace("(", "\(")
        #img = img.replace(")", "\)")
        #img = '"' + img + '"'

    input_name = create_text_path(text=text, img=img, encoding=encoding)
    
    # switch to own folder
    original_dir = os.getcwd()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    image_width = args["image_width"]
    name = os.path.join("deepdaze", str(image_width), time_str + "_" + input_name)
    os.makedirs(name, exist_ok=True)
    # copy start image to folder
    args = dict(args)
    if "start_image_path" in args:
        subprocess.run(["cp", args["start_image_path"], name])
    # copy image for feature extraction to folder
    if img is not None and isinstance(img, str):
        img_new_name = img.split("/")[-1] # only take end path
        remove_list = [")", "(", "[", "]", '"', "'"]
        for char in remove_list:
            img_new_name = img_new_name.replace(char, "")
        subprocess.run(["cp", img, os.path.join(name, img_new_name)])
        img = img_new_name
    os.chdir(name)
    # save hyperparams:
    with open("hyperparams.json", "w+") as f:
        json.dump(args, f)

    try:
        #if args["create_story"]:
        #    args["iterations"] = 2100
        imagine = Imagine(
            text=text,
            img=img,
            clip_encoding=encoding,

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
        file_names = '"' + input_name + ".000%03d.jpg" + '"'
        movie_name = '"' + input_name + ".mp4" + '"'
        subprocess.run(" ".join(["ffmpeg", "-i", file_names, "-pix_fmt", "yuv420p", movie_name]), shell=True)
        # save
        del imagine.perceptor
        del imagine.model.perceptor
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
parser.add_argument("--epochs", default=8, type=int)
parser.add_argument("--story_start_words", default=5, type=int)
parser.add_argument("--story_words_per_epoch", default=5, type=int)

# for 512: 
    # bs==1,  num_layers==22 - 7.96 GB
    # bs==2,  num_layers==20 - 7.5 GB
    # bs==16, num_layers==16 - 6.5 GB

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

args["batch_size"] = 32
args["epochs"] = 4
args["lower_bound_cutout"] = 0.01

lama = "A llama wearing a scarf and glasses, reading a book in a cozy cafe."
wizard = "A wizard in blue robes is painting a completely red image in a castle."
consciousness = "Consciousness"

bathtub_love = "I love you like a bathtub full of ice cream."
bathtub = "A bathtub full of ice cream."

prompt = lama


args["center_bias"] = True
args["center_focus"] = 1

args["image_width"] = 512
args["batch_size"] = 8
args["num_layers"] = 20

run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The ocean.", args=args)
run(text="Magma.", args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["image_width"] = 256
args["batch_size"] = 96
args["num_layers"] = 44

run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The ocean.", args=args)
run(text="Magma.", args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)


quit()


run(text=bathtub, args=args, center_bias=True, center_focus=1, epochs=7)
run(text=bathtub_love, args=args, center_bias=True, center_focus=1, epochs=7)

run(text=bathtub, args=args, center_bias=True, center_focus=3, epochs=7)
run(text=bathtub_love, args=args, center_bias=True, center_focus=3, epochs=7)

run(text=bathtub, args=args, center_bias=True, center_focus=10, epochs=7)
run(text=bathtub_love, args=args, center_bias=True, center_focus=10, epochs=7)

run(text=bathtub, args=args, center_bias=True, center_focus=2, epochs=7, optimizer="DiffGrad")
run(text=bathtub_love, args=args, center_bias=True, center_focus=2, epochs=7, optimizer="DiffGrad")

run(text=bathtub, args=args, center_bias=True, center_focus=2, epochs=7, avg_feats=True)
run(text=bathtub_love, args=args, center_bias=True, center_focus=2, epochs=7, avg_feats=True)

quit()

run(text=wizard, args=args, center_bias=True, center_focus=1, epochs=7)
run(text=prompt, args=args, center_bias=True, center_focus=1, epochs=7, gauss_mean=1.0, gauss_std=0.2, gauss_sampling=True)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, gauss_mean=0.8, gauss_std=0.2, gauss_sampling=True)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, gauss_mean=0.6, gauss_std=0.2, gauss_sampling=True)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, optimizer="Adam")
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, optimizer="DiffGrad")
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, do_occlusion=True)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, avg_feats=True)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, average_over_noise=True, noise_std=0.01, noise_n=3)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, average_over_noise=True, noise_std=0.01, noise_n=3, gauss_mean=0.6, gauss_std=0.2)
run(text=prompt, args=args, center_bias=True, center_focus=2, epochs=7, average_over_noise=True, noise_std=0.01, noise_n=3, gauss_mean=0.6, gauss_std=0.2, do_occlusion=True)



# test resnets
args["lr"] = 7e-6
args["batch_size"] = 16

run(text=prompt, args=args, model_name="RN50")
run(text=prompt, args=args, model_name="RN101")
run(text=prompt, args=args, model_name="RN50x4")

quit()

run(text=prompt, args=args)
run(text=prompt, args=args, center_bias=True, center_focus=2)
run(text=prompt, args=args, center_bias=True, center_focus=5)
run(text=prompt, args=args, center_bias=True, center_focus=10, epochs=7)
run(text=prompt, args=args, center_bias=True, center_focus=10, epochs=7, gauss_mean=0.6, gauss_std=0.2, gauss_sampling=True)
run(text=prompt, args=args, center_bias=True, center_focus=10, epochs=7, optimizer="Adam")
run(text=prompt, args=args, center_bias=True, center_focus=10, epochs=7, optimizer="DiffGrad")

# test resnets
args["lr"] = 7e-6
args["batch_size"] = 16

run(text=prompt, args=args, model_name="RN50")
run(text=prompt, args=args, model_name="RN101")
run(text=prompt, args=args, model_name="RN50x4")


quit()

# test ViT
args["model_name"] = "ViT-B/32"
args["batch_size"] = 96

# test resnets
args["lr"] = 7e-6
args["batch_size"] = 16

run(text=prompt, args=args, model_name="RN50")
run(text=prompt, args=args, model_name="RN101")
run(text=prompt, args=args, model_name="RN50x4")

prompt = consciousness

run(text=prompt, args=args, center_bias=True, center_focus=2)
run(text=prompt, args=args, center_bias=True, center_focus=5)
run(text=prompt, args=args, center_bias=True, center_focus=10)
run(text=prompt, args=args, center_bias=True, gauss_mean=0.6, gauss_std=0.2, gauss_sampling=True)
run(text=prompt, args=args, optimizer="DiffGrad")
run(text=prompt, args=args, optimizer="Adam")
run(text=prompt, args=args, optimizer="AdamP")




quit()

args["lr"] = 5e-6


# test optimizers
args["model_name"] = "ViT-B/32"

args["optimizer"] = "DiffGrad"
run(text="A wizard painting a completely red image.", **args)

args["optimizer"] = "Adam"
run(text="A wizard painting a completely red image.", **args)

args["optimizer"] = "AdamP"
run(text="A wizard painting a completely red image.", **args)


# test Resnets
args["optimizer"] = "AdamP"

# LR too high??
args["model_name"] = "RN50"
run(text="A wizard painting a completely red image.", **args)

args["model_name"] = "RN101"
run(text="A wizard painting a completely red image.", **args)

args["model_name"] = "RN50x4"
run(text="A wizard painting a completely red image.", **args)







quit()

args["center_bias"] = True
# uniform
run(text=prompt, **args)
# avg feats
args["avg_feats"] = True
run(text=prompt, **args)
args["avg_feats"] = False
# gauss size
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.2
run(text=prompt, **args)
# occlude
args["do_occlusion"] = True
run(text=prompt, **args)

quit()

# lower lower_bound
args["lower_bound_cutout"] = 0.01
run(text=prompt, **args)
args["lower_bound_cutout"] = 0.1

# baseline, uniform sampling
args["gauss_sampling"] = False
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.2
args["do_occlusion"] = False
args["average_over_noise"] = False
args["noise_std"] = 0.01
args["noise_n"] = 3
run(text=prompt, **args)
# avg feats
args["avg_feats"] = True
run(text=prompt, **args)
args["avg_feats"] = False

# test noise averaging
args["average_over_noise"] = True
args["noise_std"] = 0.01
args["noise_n"] = 3
#run(text=prompt, **args)
args["noise_std"] = 0.01
args["noise_n"] = 10
#run(text=prompt, **args)
args["noise_std"] = 0.001
args["noise_n"] = 5
#run(text=prompt, **args)
args["noise_std"] = 0.1
args["noise_n"] = 5
#run(text=prompt, **args)
args["average_over_noise"] = False

# test occlusion
args["do_occlusion"] = True
run(text=prompt, **args)
args["average_over_noise"] = True
#run(text=prompt, **args)
args["average_over_noise"] = False
args["gauss_sampling"] = True
args["gauss_mean"] = 0.2
args["gauss_std"] = 0.6
run(text=prompt, **args)

# test gauss sampling
args["gauss_sampling"] = True
args["gauss_mean"] = 0.2
args["gauss_std"] = 0.2
run(text=prompt, **args)
args["gauss_mean"] = 0.4
args["gauss_std"] = 0.2
run(text=prompt, **args)
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.2
run(text=prompt, **args)
args["gauss_mean"] = 0.8
args["gauss_std"] = 0.2
run(text=prompt, **args)
args["gauss_mean"] = 0.2
args["gauss_std"] = 0.1
run(text=prompt, **args)
args["gauss_mean"] = 0.4
args["gauss_std"] = 0.1
run(text=prompt, **args)
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.1
run(text=prompt, **args)
args["gauss_mean"] = 0.8
args["gauss_std"] = 0.1
run(text=prompt, **args)
args["gauss_sampling"] = False

# all
args["gauss_sampling"] = True
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.2
args["do_occlusion"] = True
args["average_over_noise"] = False
run(text=prompt, **args)
# avg feats
args["avg_feats"] = True
run(text=prompt, **args)
args["avg_feats"] = False
# no occl
args["gauss_sampling"] = True
args["gauss_mean"] = 0.6
args["gauss_std"] = 0.2
args["do_occlusion"] = False
args["average_over_noise"] = False
run(text=prompt, **args)
# avg feats
args["avg_feats"] = True
run(text=prompt, **args)
args["avg_feats"] = False



quit()

args["lower_bound_cutout"] = 0.1
args["upper_bound_cutout"] = 1.0
args["do_occlusion"] = False


args["lower_bound_cutout"] = 0.5
args["avg_feats"] = False
run(text="A wizard painting a completely red image.", **args)
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args)
run(text="A man painting a completely red painting.", **args)
run(text="Shattered plates on the grass.", **args)
run(text="A demon.", **args)
run(text="A psychedelic experience on LSD", **args)
run(text="Schizophrenia", **args)
run(text="Consciousness", **args)
run(text="Depression", **args)
run(text="Red", **args)
run(text="A checkerboard pattern.", **args)
run(text="A single, naked man walking on a straight, empty road by himself.", **args)
run(text="An illustration of a cat.", **args)


args["lower_bound_cutout"] = 0.5
args["avg_feats"] = True
run(text="A wizard painting a completely red image.", **args)
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args)
run(text="A man painting a completely red painting.", **args)
run(text="Shattered plates on the grass.", **args)
run(text="A demon.", **args)
run(text="A psychedelic experience on LSD", **args)
run(text="Schizophrenia", **args)
run(text="Consciousness", **args)
run(text="Depression", **args)
run(text="Red", **args)
run(text="A checkerboard pattern.", **args)
run(text="A single, naked man walking on a straight, empty road by himself.", **args)
run(text="An illustration of a cat.", **args)

quit()

args["lower_bound_cutout"] = 0.2
args["saturate_bound"] = False
run_from_file("poems/best_poems.txt", create_story=True, **args)


quit()

args["epochs"] = 1

args["lower_bound_cutout"] = 0.1
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args) 
args["lower_bound_cutout"] = 0.5
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args)
args["lower_bound_cutout"] = 0.1
args["saturate_bound"] = True
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args)
args["lower_bound_cutout"] = 0.5
args["saturate_bound"] = True
run(text="A llama wearing a scarf and glasses, reading a book in a cozy cafe.", **args)


args["lower_bound_cutout"] = 0.1
args["saturate_bound"] = True
run(text="Mist over green hills", **args)
run(text="Shattered plates on the grass", **args)
run(text="Cosmic love and attention", **args)
run(text="A time traveler in the crowd.", **args)
run(text="Life during the plague.", **args)
run(text="Meditative peace in a sunlit forest.", **args)
run(text="A psychedelic experience on LSD", **args)

args["lower_bound_cutout"] = 0.2
args["saturate_bound"] = True
run(text="Mist over green hills", **args)
run(text="Shattered plates on the grass", **args)
run(text="Cosmic love and attention", **args)
run(text="A time traveler in the crowd.", **args)
run(text="Life during the plague.", **args)
run(text="Meditative peace in a sunlit forest.", **args)
run(text="A psychedelic experience on LSD", **args)


quit()

args["lower_bound_cutout"] = 0.1
args["saturate_bound"] = False
run_from_file("poems/best_poems.txt", create_story=True, **args)
args["lower_bound_cutout"] = 0.5
args["saturate_bound"] = False
run_from_file("poems/best_poems.txt", create_story=True, **args)
args["lower_bound_cutout"] = 0.1
args["saturate_bound"] = True


run_from_file("poems/best_poems.txt", create_story=True, **args)

quit()

run(text="Adam Ondra climbing", **args)
run(text="A climber climbing a big mountain", **args)
run(text="Rock climbing", **args)
run(text="Bouldering", **args)
run(text="Happiness", **args)
run(text="Being born", **args)
run(text="The process of dying", **args)
run(text="Life", **args)
run(text="Death", **args)
run(text="Meditation", **args)
#run(text="LSD", **args)

run_from_file("dreams_male_college.txt", create_story= True, **args)
run_from_file("dreams_female_college.txt", create_story=True, **args)


run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", **args)
run(text="Basking in sunlight.", **args)
run(text="Beauty of life.", **args)
run(text="Marvellous. Glamorous. Beautiful.", **args)
run(text="Yoga.", **args)
run(text="Meditative surfing on the crescent waves of the ocean.", **args)


run_from_file("poems/poems_10_0.txt", create_story=True, **args)
args["iterations"] = 500
run_from_file("poems/poems_10_0.txt", create_story=True, **args)


quit()
run(text="A wizard painting a completely red image.", **args)
run(text="Schizophrenia!", **args)
run(text="Depression.", **args)
run(text="Sadness.", **args)
run(text="The most beautiful painting.", **args)
run(text="The most ugly painting.", **args)


run(text="Mist over green hills", **args)
run(text="Shattered plates on the grass", **args)
run(text="Cosmic love and attention", **args)
run(text="A time traveler in the crowd.", **args)
run(text="Life during the plague.", **args)
run(text="Meditative peace in a sunlit forest.", **args)
run(text="A psychedelic experience on LSD", **args)

args["iterations"] = 500
run_from_file("dreams_female_college.txt", create_story=True, **args)
run_from_file("dreams_male_college.txt", create_story=True, **args)

quit()

# some favourites
run(text="Consciousness", **args)
run(text="Enlightenment", **args)
run(text="Depression", **args)
run(text="Multiple personality disorder", **args)
run(text="Schizophrenia", **args)

run(text="The all-seeing eye.", **args)
run(text="The all-seeing tree.", **args)
run(text="The blue all-seeing tree. A blue tree with a large eye in its stem", **args)
run(text="Being born", **args)
run(text="The process of dying", **args)

run(text="A psychedelic experience on magic mushrooms", **args)
run(text="A psychedelic experience on LSD", **args)
run(text="A psychedelic experience on Mescaline", **args)
run(text="A psychedelic experience on Salvia Divinorum", **args)

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


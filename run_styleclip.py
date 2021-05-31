import argparse
import time
import os
import subprocess
import json
import sys
import copy
import shutil
import copy

import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

sys.path.append("../StyleCLIP_modular")
from style_clip import Imagine, create_text_path


def run(text=None, img=None, encoding=None, name=None, args=None, **kwargs):
    if args is None:
        args = {}
    args = copy.copy(args)
    for key in kwargs:
        args[key] = kwargs[key]

    img_name = img.split("/")[-1] if img is not None else None
    input_name = create_text_path(text=text, img=img_name, encoding=encoding, context_length=77)
    
    # switch to own folder
    original_dir = os.getcwd()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    model_type = args["model_type"]
    if model_type == "stylegan":
        remove = ["(", " ", ")", "/", "-", "[", "]"]
        style = args["style"].split("/")[-1].split(".")[0]
        for c in remove:
            style = style.replace(c, "")
        name = os.path.join("style_clip", str(style), time_str + "_" + input_name)
    else:
        name = os.path.join(model_type, time_str + "_" + input_name)
    os.makedirs(name, exist_ok=True)
    # copy start image to folder
    args = dict(args)
    if "start_image_path" in args:
        shutil.copy(args["start_image_path"], name)
    # copy image for feature extraction to folder
    if img is not None and isinstance(img, str):
        img_new_name = img.split("/")[-1] # only take end path
        remove_list = [")", "(", "[", "]", '"', "'"]
        for char in remove_list:
            img_new_name = img_new_name.replace(char, "")
        shutil.copy(img, os.path.join(name, img_new_name))
        img = img_new_name

    try:        
        imagine = Imagine(
            save_progress=True,
            open_folder=False,
            save_video=True,
            **args
           )
        
        # copy upfir2dn file from stylegan2 to new working directory
        #upfir_path = "stylegan2/torch_utils/ops/" #upfirdn2d.cpp"
        #upfir_new_path = os.path.join(name, upfir_path)
        #os.makedirs(upfir_new_path, exist_ok=True)  #"/".join(upfir_new_path.split("/")[:-1]), exist_ok=True)
        #shutil.copy(upfir_path + "upfirdn2d.cpp", os.path.join(name, upfir_path + "upfirdn2d.cpp"))
        #shutil.copy(upfir_path + "upfirdn2d.cu", os.path.join(name, upfir_path + "upfirdn2d.cu"))

        os.chdir(name)
        
        imagine.set_clip_encoding(text=text,
            img=img,
            encoding=encoding,
        )
        
        # save hyperparams:
        save_args = copy.copy(args)
        if "transform" in save_args:
            del save_args["transform"]
        with open("hyperparams.json", "w+") as f:
            json.dump(save_args, f)
       
        sys.path.append("../../stylegan2")
        sys.path.append("../../../stylegan2")
        sys.path.append("../../../../stylegan2")

        # train
        imagine()
        
        # plot losses
        plt.figure()
        plt.plot(imagine.aug_losses, label="Augmented")
        plt.plot(imagine.non_aug_losses, label="Raw")
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.clf()
       
        # save
        del imagine.perceptor
        del imagine.model.perceptor
        #torch.save(imagine.cpu(), "model.pt")
        del imagine

    finally:
        os.chdir(original_dir)


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--gradient_accumulate_every", default=1, type=int)
parser.add_argument("--save_every", default=1, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--story_start_words", default=5, type=int)
parser.add_argument("--story_words_per_epoch", default=5, type=int)
parser.add_argument("--style", default="../stylegan2-ada-pytorch/VisionaryArt.pkl", type=str, choices=["faces (ffhq config-f)", "../stylegan2-ada-pytorch/VisionaryArt.pkl"])
parser.add_argument("--lr_schedule", default=0, type=int)
parser.add_argument("--start_image_steps", default=1000, type=int)
parser.add_argument("--iterations", default=100, type=int)


#parser.add_argument("--seed")

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


args["lower_bound_cutout"] = 0.1

lama = "A llama wearing a scarf and glasses, reading a book in a cozy cafe."
wizard = "A wizard in blue robes is painting a completely red image in a castle."
consciousness = "Consciousness"
bathtub_love = "I love you like a bathtub full of ice cream."
bathtub = "A bathtub full of ice cream."
snail = "A vector illustration of a snail made of harp. A snail with the texture of a harp."
phoenix = "A phoenix rising from the ashes."

# choose style
#args["style"] = "../lucid-sonic-dreams/faces (ffhq config-f).pkl"
#args["style"] = "../stylegan2-ada-pytorch/logos_17040.pkl"
#args["style"] = "../stylegan2-ada-pytorch/madziowa_p_800.pkl"
#args["style"] = "../stylegan2-ada-pytorch/jan_regnart_640.pkl"
#args["style"] = "../stylegan2-ada-pytorch/astromaniacmag_160.pkl"
args["style"] = "../stylegan2-ada-pytorch/vint_retro_scifi_3200_2map.pkl"



args["opt_all_layers"] = 1
args["lr_schedule"] = 1
args["noise_opt"] = 0
args["reg_noise"] = 0
args["seed"] = 1

# Exps:

dalle_prompts = ["an armchair in the shape of an avocado, an armchair imitating an avocado", "A painting in cubist style of a capybara sitting in a forest at sunrise", "a lamp in the shape of a pikachu, a lamp imitating a pikachu", "a fox made of voxels sitting on a mountain"]


args["style"] = "../stylegan2-ada-pytorch/vint_retro_scifi_3200_2map.pkl"

#run(text="An alien", args=args, iterations=50)

args["model_type"] = "vqgan"
args["save_every"] = 20
args["start_img_loss_weight"] = 0.0
args["start_image_steps"] = 500
args["lr"] = 0.1
args["sideX"] = 480
args["sideY"] = 480
args["batch_size"] = 32
args["iterations"] = 1500
# codebook_size = 1024, # [1024, 16384]


bands_that_arent_real = ["Necrotic Tyrant - Lost Crypts of the Unremembered", "Apocalyptic Maelstrom - Darkwater Monoliths", "Of Ancient Times - Necropolis of the Blood Red Sun", "Sectomancer - Blood from a Dying Sun", "Beneath the Dark Sea - It Lurks in the Deep"]
prank_comics = ["Take a picture of your bathroom and plaster it on your fridge.", "Stand in line for a movie for 30 minutes then leave.", "Ordering junk treats from TV ads at 3 in the morning, like those 'falling in the ocean' doughnuts", "Eat a banan inside your own mouth", "Have a tortoise deliver your package for you, don't be surprised if it sings 'Happy Birthday' back at you.", "Opt for the treadmill. You'll be running in the nude.", "Put a cup of coffee on your lap. It's an oldie, but it's still a favourite.", "My cat slept through a December blizzard in Florida this year", "There's a square of chicken in your front yard", "Placing a crown on your head."]
pinar_1 = ["quantum physics", "God", "The soul of the world", "unconditional love", "Spiritual teacher", "Spirituality", "Goddess of the world", "Anima Mundi", "Twilight state", "Children of god", "mama matrix most mysterious", "psychic", "the power of magic", "psilocybin", "shamanism", "non-human intelligence", "alchemy", "Amongst the elves", "non-duality", "duality", "metaphysical sensitivity"]

def add_context(words, prefix="", suffix=""):
    return [prefix + word + suffix for word in words]



args["sideX"] = 1280
args["sideY"] = 720
args["batch_size"] = 32
args["iterations"] = 500

run(text="David Bowie on the moon", args=args)

quit()

args["optimizer"] = "Adam"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["optimizer"] = "AdamP"

args["optimizer"] = "DiffGrad"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["optimizer"] = "AdamP"



for prompt in pinar_1:
    run(text=prompt, args=args, circular=1)
    
for prompt in pinar_1:
    run(text=prompt, args=args, circular=0)
    

quit()

args["circular"] = 1
run(text="AdaLab x SeyhanLee", args=args)
run(text="We are on the cusp of a new era", args=args)
run(text="Sunday worship", args=args)
run(text="A new era", args=args)
run(text="Genesis", args=args)


quit()

args["circular"] = 1
run(text="Consciousness", args=args, iterations=500)
run(text="Consciousness", args=args)
run(text="David Bowie", args=args)
run(text="Shia LaBouef", args=args)
run(text="SeyhanLee", args=args)
run(text="AdaLab", args=args)

quit()

for band in add_context(bands_that_arent_real, suffix=", a death metal album cover."):
    run(text=band, args=args, neg_text="incoherent, written text")

for band in add_context(bands_that_arent_real, suffix=", a death metal album cover."):
    run(text=band, args=args)



args["latent_type"] = "code_sampling"

for band in bands_that_arent_real:
    run(text=band, args=args)
    
for band in add_context(bands_that_arent_real, prefix="An album cover for: "):
    run(text=band, args=args)
    
for band in add_context(bands_that_arent_real, suffix=", an album cover."):
    run(text=band, args=args)
    
for band in add_context(bands_that_arent_real, suffix=", an death metal album cover."):
    run(text=band, args=args)

for prank in prank_comics:
    run(text=prank, args=args)
    
for prank in add_context(prank_comics, prefix="A black-and-white comic displaying: "):
    run(text=prank, args=args)
    
for prank in add_context(prank_comics, suffix=", illustrated as a comic."):
    run(text=prank, args=args)
    
for prank in add_context(prank_comics, suffix=", illustrated with stick figures."):
    run(text=prank, args=args)
    


quit()

neg_text = '''incoherent, confusing, cropped, watermarks'''
args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["neg_text"] = None


run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

quit()

args["latent_type"] = "embedding_sinh"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["latent_type"] = "embedding"


args["clip_latents"] = 1
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["clip_latents"] = 0

args["optimizer"] = "AdamW"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["optimizer"] = "AdamP"

args["codebook_size"] = 16384
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["codebook_size"] = 16384
args["latent_type"] = "code_sampling"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["codebook_size"] = 1024
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

neg_text = '''incoherent, confusing, cropped, watermarks'''
args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)


args["latent_type"] = "embedding"
args["noise_augment"] = 1
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

quit()


args["lr_schedule"] = 0
run(text=phoenix, args=args, iterations=2000)

args["lr_schedule"] = 1
run(text=phoenix, args=args, iterations=10)
run(text=phoenix, args=args, iterations=50)
run(text=phoenix, args=args, iterations=100)
run(text=phoenix, args=args, iterations=200)
run(text=phoenix, args=args, iterations=500)
run(text=phoenix, args=args, iterations=1000)
run(text=phoenix, args=args, iterations=2000)
run(text=phoenix, args=args, iterations=5000)
run(text=phoenix, args=args, iterations=10000)






quit()

args["sideX"] = 360
args["sideY"] = 360
args["lr_schedule"] = 0
args["lr"]= 0.05
args["save_every"] = 2



args["iterations"] = 50
run(start_image_path="base_images/michi_san_diego.jpg", text="Green mist on the highway", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Green", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Green jungle", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Jungle", args=args)

args["start_img_loss_weight"] = 0.1
args["iterations"] = 50
run(start_image_path="base_images/michi_san_diego.jpg", text="Green mist on the highway", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Green", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Green jungle", args=args)
run(start_image_path="base_images/michi_san_diego.jpg", text="Jungle", args=args)

quit()


args["iterations"] = 2000
args["save_every"] = 5
run(start_image_path="base_images/michi_san_diego.jpg", img="base_images/michi_san_diego.jpg", args=args)


quit()

run(img="base_images/michi_san_diego.jpg", args=args)


args["latent_type"] = "code_sampling"
run(img="base_images/ouzi.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)
run(img="base_images/michi_san_diego.jpg", args=args)

quit()

run(text=snail, args=args)
run(text=snail, args=args, latent_type="code_sampling")
run(text=snail, args=args, use_tv_loss=True)
run(img="base_images/earth.jpg", args=args)
run(img="base_images/ouzi.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)
run(img="base_images/hot-dog.jpg", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/stance.jpg", args=args)


quit()

run(text="A fusion reactor", args=args)
run(text="Fusion", args=args)
run(text="Fusion plasma", args=args)
run(text="An artificial sun", args=args)
run(text="A lab-grown sun", args=args)
run(text="The all mighty power of the sun", args=args)
run(text="The solution of the climate crisis", args=args)
run(text="A healthy and sustainable meal", args=args)
run(text="The effects of the climate change", args=args)
run(text="The effects of the climate crisis", args=args)
run(text="Apocalypse", args=args)
run(text="Life during a pandemic", args=args)
run(text="Lockdown", args=args)

quit()

args["use_tv_loss"] = 0
run(text="Extinction Rebellion", args=args)
run(text="Happy Birthday", args=args)
run(text="Alles gute zum Geburtstag", args=args)
run(text="Feliz Cumpleanos", args=args)
run(text="Alles gute zum Geburtstag, Michi, du alter Racker!", args=args)
run(text="Happy birthday Michi, you old rascal!", args=args)

args["use_tv_loss"] = 1
run(text="Extinction Rebellion", args=args)
run(text="Happy Birthday", args=args)
run(text="Alles gute zum Geburtstag", args=args)
run(text="Feliz Cumpleanos", args=args)
run(text="Alles gute zum Geburtstag, Michi, du alter Racker!", args=args)
run(text="Happy birthday Michi, you old rascal!", args=args)



quit()



run(text="A black hole", args=args)
run(text="Artificial Intelligence", args=args)
run(text="Big city life", args=args)
run(text="The most beautiful painting ever", args=args)
run(text="The most ugly painting ever", args=args)
run(text="A fire tornado that shoots lasers at a dinosaur", args=args)
run(text="An apocalyptic train", args=args)
run(text="Theorem of Pythagoras", args=args)
run(text="Mathematics", args=args)
run(text="Physics", args=args)
run(text="Jupyter's moons", args=args)
run(text="Pure love and acceptance", args=args)
run(text="Nazis riding dinosaurs while shooting lasers", args=args)
run(text="Space pussy", args=args)
run(text="The meaning of life", args=args)
run(text="What is the meaning of life?", args=args)
run(text="The useless third of the population", args=args)
run(text="Freedom after an eternal lockdown", args=args)
run(text="Freedom by vaccination", args=args)
run(text="Hope", args=args)
run(text="Kartoffelsalat", args=args)
run(text="Seeing without your eyes", args=args)
run(text="Weltschmerz", args=args)
run(text="Wanderlust", args=args)
run(text="Innocence", args=args)
run(text="A warm summer breeze", args=args)
run(text="The smell of spring", args=args)
run(text="Silky clouds in the sky that paint my carefree feelings", args=args)
run(text="Underground rebels hiding from a train", args=args)
run(text="The matrix", args=args)
run(text="Are we living in a simulation?", args=args)
run(text="Rescuing queers from a desert prison with a jeep", args=args)
run(text="2 + 2", args=args)
run(text="1 + 1", args=args)


quit()

run(text="Digital music stored on woven fabric by people dancing around a pyramid.", args=args)
run(text="Digital music stored on woven fabric by people on drugs dancing around a pyramid.", args=args)
run(text="Digital music stored on woven fabric by people on drugs dancing around a fire pyramid.", args=args)
run(text="Digital music stored on woven fabric.", args=args)
run(text="Digital music stored on colourful woven fabric.", args=args)
run(text="A computer historian showing off his collection in music.", args=args)
run(text="An angry red giant rubber duck attacking a group of hunters.", args=args)
run(text="A hunting party attract an angry red giant rubber duck.", args=args)
run(text="Micheal Büchler", args=args)
run(text="Anton Wiehe", args=args)
run(text="Nadia Burke", args=args)
run(text="Thomas Hamacher", args=args)
run(text="A lightning shimmering over the green ocean", args=args)
run(text="Ultra cold plasma", args=args)

args["latent_type"] = "code_sampling"
run(text="Digital music stored on woven fabric by people dancing around a pyramid.", args=args)
run(text="Digital music stored on woven fabric by people on drugs dancing around a pyramid.", args=args)
run(text="Digital music stored on woven fabric by people on drugs dancing around a fire pyramid.", args=args)
run(text="Digital music stored on woven fabric.", args=args)
run(text="Digital music stored on colourful woven fabric.", args=args)
run(text="A computer historian showing off his collection in music.", args=args)
run(text="An angry red giant rubber duck attacking a group of hunters.", args=args)
run(text="A hunting party attract an angry red giant rubber duck.", args=args)
run(text="Micheal Büchler", args=args)
run(text="Anton Wiehe", args=args)
run(text="Nadia Burke", args=args)
run(text="Thomas Hamacher", args=args)
run(text="A lightning shimmering over the green ocean", args=args)
run(text="Ultra cold plasma", args=args)


quit()


run(text="Anton going into a rabbit hole", args=args)
run(text="Opening a door to parallel dimensions", args=args)

args["latent_type"] = "code_sampling"
run(text="Anton going into a rabbit hole", args=args)
run(text="Opening a door to parallel dimensions", args=args)


quit()

run(text="The mushroom wizard", args=args)
run(text="The fire wizard", args=args)
run(text="Fire fractal", args=args)
run(text="High quality HD rendering of fire", args=args)
run(text="HD rendering of water", args=args)


args["latent_type"] = "code_sampling"
run(text="The mushroom wizard", args=args)
run(text="The fire wizard", args=args)
run(text="Fire fractal", args=args)
run(text="High quality HD rendering of fire", args=args)
run(text="HD rendering of water", args=args)


quit()

run(text="Meditative peace in a sunlit forest", args=args)
run(text="Shattered plates on the grass", args=args)
run(text="Fire in the sky", args=args)
run(text="The mushroom wizard", args=args)

quit()

args["latent_type"] = "code_sampling"

run(text="The mushroom wizard", args=args)
run(text="The first supper", args=args)
run(text="The last supper", args=args)


quit()

run(text="Anton", args=args)
run(text="Anton Wiehe", args=args)
run(text="Meditative peace in a sunlit forest", args=args)
run(text="Shattered plates on the grass", args=args)
run(text="Fire in the sky", args=args)
run(text="The mushroom wizard", args=args)



quit()

run(text="She was on a film set with psychedelic aliens", args=args)
args["latent_type"] = "code_sampling"


run(text="Eternity looped forever", args=args)
run(text="Existence consciousness integration", args=args)
run(text="The mushroom wizard", args=args)
run(text="She was on a film set with psychedelic aliens", args=args)


quit()

run(text="The library of the sun", args=args)
run(text="Intricate nothing", args=args)


quit()

run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["latent_type"] = "code_sampling"
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["use_tv_loss"] = 1
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["transform"] = 0
run(text="David Bowie", args=args)
run(text=lama, args=args)
run(text=wizard, args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)


quit()


args["latent_type"] = "code_sampling"
run(text="David Bowie", args=args)
run(text=lama, args=args)
run(text=wizard, args=args)

quit()

args["use_tv_loss"] = 1
run(text="David Bowie", args=args)
run(text=lama, args=args)
run(text=wizard, args=args)

quit()

run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)


quit()

args["latent_type"] = "embedding" # embedding, code, sampled_embedding,
#run(text="David Bowie", args=args)
#run(text=lama, args=args)
#run(text=wizard, args=args)

# does not work
#args["latent_type"] = "code" # embedding, code, sampled_embedding,
#run(text="David Bowie", args=args)
#run(text=lama, args=args)
#run(text=wizard, args=args)

args["latent_type"] = "sampled_embedding" # embedding, code, sampled_embedding,
run(text="David Bowie", args=args)
run(text=lama, args=args)
run(text=wizard, args=args)


quit()
args["averaging_weight"] = 0.0
#run(text="Love", args=args)
#run(text="David Bowie.", args=args)
#run(text="Death.", args=args)
#run(text="Schizophrenia.", args=args)
#run(text="A psychedelic experience on LSD.", args=args)
#run(text="Consciousness.", args=args)

args["averaging_weight"] = 1.0
#run(text="Love", args=args)
#run(text="David Bowie.", args=args)
#run(text="Death.", args=args)
#run(text="Schizophrenia.", args=args)
#run(text="A psychedelic experience on LSD.", args=args)
#run(text="Consciousness.", args=args)


# fully new custom transform
color = 0.9
degrees = 10
transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.1, 1.0)),#, ratio=(0.75, 1.3333333333333333)),
    T.ColorJitter(brightness=color, contrast=color, saturation=color, hue=color / 8),
    T.RandomAffine(10, translate=(0.01, 0.2), shear=None),
    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])
args["transform"] = transform
args["do_cutout"] = 0
args["averaging_weight"] = 0.0
#run(text="Love", args=args)
#run(text="David Bowie.", args=args)
#run(text="Death.", args=args)
#run(text="Schizophrenia.", args=args)
#run(text="A psychedelic experience on LSD.", args=args)
#run(text="Consciousness.", args=args)

# new transform without cutouts!
transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ColorJitter(brightness=color, contrast=color, saturation=color, hue=color / 8),
    T.RandomAffine(10, translate=(0.01, 0.2), shear=None),
    T.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
])
args["transform"] = transform
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)
run(text="Schizophrenia.", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="Consciousness.", args=args)

quit()

# Dalle-exp
for prompt in dalle_prompts:
    run(text=prompt, args=args)
    
quit()


args["sideX"] = 128
args["sideY"] = 128
args["batch_size"] = 8
# bs8xga8  7GB
# batch size 64
args["gradient_accumulate_every"] = 8
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)


args["batch_size"] = 32


run(text="The first supper", args=args)
run(text="The last supper", args=args)
run(text="The first supper", args=args, circular=1)
run(text="The last supper", args=args, circular=1)

quit()


args["batch_size"] = 32
args["transform"] = 0
args["circular"] = 0
#run(text="Good old regular cut-out transforms.", args=args)
#run(text="Love", args=args)
#run(text="David Bowie.", args=args)
#run(text="Death.", args=args)
#run(text="Schizophrenia.", args=args)
#run(text="A psychedelic experience on LSD.", args=args)
#run(text="Consciousness.", args=args)


quit()

args["transform"] = None
args["circular"] = 0
run(text="Old cut-outs with new fancy transforms that center the object.", args=args)
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)
run(text="Schizophrenia.", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="Consciousness.", args=args)

args["transform"] = None
args["circular"] = 1
run(text="Weird new circular images", args=args)
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)
run(text="Schizophrenia.", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="Consciousness.", args=args)


args["batch_size"] = 2
args["transform"] = None
args["circular"] = 0
run(text="Batch size tests.", args=args)
run(text="Love", args=args)

args["batch_size"] = 4
run(text="Love", args=args)

args["batch_size"] = 8
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)

args["batch_size"] = 16
run(text="Love", args=args)
run(text="David Bowie.", args=args)
run(text="Death.", args=args)


quit()

run(img="base_images/aicpa_logo_black.jpg", start_image_path="base_images/stance.jpg", args=args)

#run(img="base_images/aicpa_logo_black.jpg", start_image_path="base_images/earth.jpg", args=args)
#run(img="base_images/earth.jpg", start_image_path="base_images/aicpa_logo_black.jpg", args=args)



quit()

run(text="A pink alien that is flying to Saturn.", args=args, center_bias=5, averaging_weight=0.8)
run(text="A pink alien that is flying to Saturn.", args=args, center_bias=5, averaging_weight=1.0)

quit()

run(text="Extravagant displayal of beauty.", args=args)
run(text="you've beaten shia labeouf", args=args)
run(text="you limp into the dark woods", args=args)
run(text="wait! he isn't dead! Shia surprise!", args=args)


quit()

run(text="Intricate nothing.", args=args)
run(text="Magic variables.", args=args)
run(text="A pink alien that is flying to Saturn.", args=args)

run(text="Troopers", args=args)
run(text="Strange planets", args=args)
run(text="Astronauts are having problems", args=args)
run(text="A monster is eating people", args=args)
run(text="Beautiful", args=args)


run(text="Intricate nothing.", args=args, center_bias=5)
run(text="Intricate nothing.", args=args, averaging_weight=0.8)

run(text="A pink alien that is flying to Saturn.", args=args, center_bias=5)
run(text="A pink alien that is flying to Saturn.", args=args, averaging_weight=0.8)





quit()

run(text="David Bowie.", args=args)
run(text="Death.", args=args)
run(text="Schizophrenia.", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="Consciousness.", args=args)
run(text="A naked lady.", args=args)
run(text="Anton.", args=args)


quit()

run(text="A green highway exit sign on the side of a desert road seen from a moving car.", args=args)
run(text="A man and a woman in love.", args=args)
run(text="Sumer Temple.", args=args)
run(text="Crab nebula.", args=args)
run(text="Shining stars.", args=args)



quit()

args["lr_schedule"] = 0

run(text="Spaceships approaching earth", args=args)
run(text="Mythical beings.", args=args, iterations=100)
run(text="Love.", args=args, iterations=100)
run(text="A green highway exit sign.", args=args, iterations=100)

args["lr_schedule"] = 1
args["iterations"] = 1000
run(text="Spaceships approaching earth", args=args)
run(text="Mythical beings.", args=args, iterations=100)
run(text="Love.", args=args, iterations=100)
run(text="A green highway exit sign.", args=args, iterations=100)



quit()


run(text="Troopers", args=args)
run(text="Strange planets", args=args)
run(text="Astronauts are having problems", args=args)
run(text="A monster is eating people", args=args)
run(text="Beautiful", args=args)


args["iterations"] = 1000
for t in dalle_prompts:
    run(text=t, args=args)

quit()

args["style"] = "../stylegan2-ada-pytorch/madziowa_p_800.pkl"
for t in dalle_prompts:
    run(text=t, args=args)

    

quit()


args["iterations"] = 2000
args["save_every"] = 4
run(text="Consciousness", args=args)
run(text="do unto others as you would have them do unto you", args=args)
run(text="Looking back", args=args)
run(text="Meditative peace in a sunlit forest.", args=args)
run(text="Gödel, Escher, Bach", args=args)
run(text="Supernova", args=args)
run(text="Ecstacy", args=args)
run(text="Boundless ego", args=args)
run(text="Oceanic boundlessness", args=args)
run(text="Home", args=args)
run(text="Wanderlust", args=args)

args["style"] = "../stylegan2-ada-pytorch/madziowa_p_800.pkl"
run(text="Consciousness", args=args)
run(text="do unto others as you would have them do unto you.", args=args)
run(text="Looking back", args=args)
run(text="Meditative peace in a sunlit forest.", args=args)
run(text="Gödel, Escher, Bach", args=args)
run(text="Supernova", args=args)
run(text="Ecstacy", args=args)
run(text="Boundless ego", args=args)
run(text="Oceanic boundlessness", args=args)
run(text="Home", args=args)
run(text="Wanderlust", args=args)


quit()


run(start_image_path="base_images/anton_climb.jpg", text="A picture of a happy man.", args=args)


run(text="A black hole", args=args)
run(text="A picture of a black hole", args=args)
run(text="A supernova", args=args)
run(text="A quasar", args=args)
run(text="A picture of a friendly old man", args=args)
run(text="Consciousness", args=args)
run(text="Depression", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="A sunflower", args=args)



args["iterations"] = 1000

run(text="A black hole", args=args)
run(text="A picture of a black hole", args=args)
run(text="A supernova", args=args)
run(text="A quasar", args=args)
run(text="A picture of a friendly old man", args=args)
run(text="Consciousness", args=args)
run(text="Depression", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="A sunflower", args=args)

quit()

run(start_image_path="base_images/anton_climb.jpg", text="A picture of a happy man.", args=args)

args["norm_weight"] = 0.1
run(start_image_path="base_images/anton_climb.jpg", text="A picture of a sad man.", args=args)

args["norm_weight"] = 1.0
run(start_image_path="base_images/anton_climb.jpg", text="A picture of a sad man.", args=args)

args["norm_weight"] = 5.0
run(start_image_path="base_images/anton_climb.jpg", text="A picture of a sad man.", args=args)


quit()


args["opt_all_layers"] = 0
args["noise_opt"] = 1
run(start_image_path="base_images/anton_climb.jpg", text="Mad man.", args=args)

args["opt_all_layers"] = 0
args["noise_opt"] = 1
args["reg_noise"] = 1
run(start_image_path="base_images/anton_climb.jpg", text="Mad man.", args=args)

args["opt_all_layers"] = 0
args["noise_opt"] = 0
args["reg_noise"] = 0
run(start_image_path="base_images/anton_climb.jpg", text="Mad man.", args=args)

args["opt_all_layers"] = 1
args["norm_weight"] = 0.2
run(start_image_path="base_images/anton_climb.jpg", text="Mad man.", args=args)

args["opt_all_layers"] = 1
args["norm_weight"] = 0.8
run(start_image_path="base_images/anton_climb.jpg", text="Mad man.", args=args)

quit()

run(text="A painting of a sunflower.", args=args)

args["opt_all_layers"] = 0
run(text="A painting of a sunflower.", args=args)

args["noise_opt"] = 1
run(text="A painting of a sunflower.", args=args)


args["opt_all_layers"] = 1
args["noise_opt"] = 0

args["batch_size"] = 64
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

args["batch_size"] = 64
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

quit()

args["batch_size"] = 64
args["lr"] = 0.1
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)


args["batch_size"] = 32
args["lr"] = 0.1
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

args["norm_weight"] = 0.1
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

args["norm_weight"] = 0.2
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

args["norm_weight"] = 0.5
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)

args["norm_weight"] = 1.0
run(text="Soft.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Techno.", args=args)




quit()


args["lr"] = 0.5
run(text="Wild", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Wild.", args=args)

args["lr"] = 0.2
run(text="Wild", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Wild.", args=args)

args["start_image_steps"] = 1000
args["lr"] = 0.1
run(start_image_path="base_images/anton_climb.jpg", text="Wild.", args=args)

args["lr"] = 0.01
run(start_image_path="base_images/anton_climb.jpg", text="Wild.", args=args)

quit()

args["lr"] = 0.1
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.05
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.02
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.01
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.005
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.1
args["lr_schedule"] = 1
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.05
args["lr_schedule"] = 1
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

args["lr"] = 0.01
args["lr_schedule"] = 1
run(text="Classical", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Classical.", args=args)

quit()

args["opt_all_layers"] = 0
args["lr_schedule"] = 0
args["noise_opt"] = 1
args["reg_noise"] = 1
run(text="Hiphop", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a person with glasses.", args=args)

args["opt_all_layers"] = 0
args["lr_schedule"] = 1
args["noise_opt"] = 1
args["reg_noise"] = 1
run(text="Hiphop", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a person with glasses.", args=args)

args["opt_all_layers"] = 1
args["lr_schedule"] = 1
args["noise_opt"] = 1
args["reg_noise"] = 1
run(text="Hiphop", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a person with glasses.", args=args)

args["start_image_steps"] = 1000
args["opt_all_layers"] = 1
args["lr_schedule"] = 1
args["noise_opt"] = 0
args["reg_noise"] = 0
run(text="Hiphop", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a person with glasses.", args=args)

args["opt_all_layers"] = 1
args["lr_schedule"] = 0
args["noise_opt"] = 0
args["reg_noise"] = 0
run(text="Hiphop", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a person with glasses.", args=args)

quit()

args["lr_schedule"] = 1
args["noise_opt"] = 0
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

args["lr_schedule"] = 1
args["noise_opt"] = 1
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

args["opt_all_layers"] = 1
args["lr_schedule"] = 1
args["noise_opt"] = 0
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

args["lr_schedule"] = 0
args["noise_opt"] = 0
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

args["lr_schedule"] = 0
args["noise_opt"] = 1
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

args["opt_all_layers"] = 1
args["lr_schedule"] = 0
args["noise_opt"] = 0
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.25
run(text="Slow. Dramatic.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a sad person.", args=args)

quit()

args["opt_all_layers"] = 1
args["noise_opt"] = 0
args["lr_schedule"] = 0
args["reg_noise"] = 0
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)

args["opt_all_layers"] = 0
args["noise_opt"] = 0
args["lr_schedule"] = 0
args["reg_noise"] = 0
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)

args["opt_all_layers"] = 0
args["noise_opt"] = 1
args["lr_schedule"] = 1
args["reg_noise"] = 0
args["start_img_loss_weight"] = 0.01
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["start_img_loss_weight"] = 0.2
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["start_img_loss_weight"] = 0.5
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["start_img_loss_weight"] = 1.0
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)

quit()


args["noise_opt"] = 0
args["lr_schedule"] = 0
args["reg_noise"] = 0
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)

args["noise_opt"] = 0
args["lr_schedule"] = 1
args["reg_noise"] = 0
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)

args["lr_schedule"] = 0
args["noise_opt"] = 1
args["reg_noise"] = 0
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["noise_opt"] = 1
args["reg_noise"] = 1
run(text="Fast. Happy.", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)


quit()

run(text="A charcoal drawing of a creepy face.", args=args)
run(text="A wirey drawing of a creepy face.", args=args)
run(text="A charcoal drawing of a cat", args=args)
run(text="A drawing of a beautiful lady.", args=args)
run(text="A drawing of a beautiful man.", args=args)
run(text="Schizophrenia", args=args)
run(text="Depression", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="A scary raven.", args=args)
run(text="A pink cat.", args=args)


run(text="Fast", args=args)
run(text="Happy", args=args)
run(text="Fast. Happy.", args=args)
run(text="Slow.", args=args)
run(text="Sad.", args=args)
run(text="Vibrant.", args=args)
run(text="Piano.", args=args)
run(text="Violin.", args=args)

args["style"] = "../stylegan2-ada-pytorch/jan_regnart_640.pkl"
run(text="Fast", args=args)
run(text="Happy", args=args)
run(text="Fast. Happy.", args=args)
run(text="Slow.", args=args)
run(text="Sad.", args=args)
run(text="Vibrant.", args=args)
run(text="Piano.", args=args)
run(text="Violin.", args=args)

quit()



args["seed"] = 1
args["start_image_steps"] = 1000

args["noise_opt"] = 0
args["reg_noise"] = 0
args["lr_schedule"] = 0
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["noise_opt"] = 0
args["reg_noise"] = 0
args["lr_schedule"] = 1
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["noise_opt"] = 1
args["reg_noise"] = 0
args["lr_schedule"] = 0
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["noise_opt"] = 1
args["reg_noise"] = 1
args["lr_schedule"] = 0
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)
args["noise_opt"] = 1
args["reg_noise"] = 1
args["lr_schedule"] = 1
run(start_image_path="base_images/anton_climb.jpg", text="A photo of a weird person.", args=args)


quit()

run(start_image_path="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", text="Beauty", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Scary", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)

args["noise_opt"] = 1
args["lr_schedule"] = 0
run(start_image_path="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", text="Beauty", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Scary", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)

args["noise_opt"] = 0
args["lr_schedule"] = 1
run(start_image_path="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", text="Beauty", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Scary", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)

args["noise_opt"] = 1
args["lr_schedule"] = 1
run(start_image_path="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", text="Beauty", args=args)
run(start_image_path="base_images/anton_climb.jpg", text="Scary", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)


args["noise_opt"] = 0
args["lr_schedule"] = 0
run(text="A charcoal drawing of a creepy face.", args=args)
run(text="A wirey drawing of a creepy face.", args=args)
run(text="A charcoal drawing of a cat", args=args)
run(text="A drawing of a beautiful lady.", args=args)
run(text="A drawing of a beautiful man.", args=args)
run(text="Schizophrenia", args=args)
run(text="Depression", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="A scary raven.", args=args)
run(text="A pink cat.", args=args)


args["style"] = "../stylegan2-ada-pytorch/jan_regnart_640.pkl"
run(text="Fast", args=args)
run(text="Happy", args=args)
run(text="Fast. Happy.", args=args)
run(text="Slow.", args=args)
run(text="Sad.", args=args)
run(text="Vibrant.", args=args)
run(text="Piano.", args=args)
run(text="Violin.", args=args)



args["noise_opt"] = 1
run(text="A mad man.", args=args)
run(text="Artificial Intelligence.", args=args)
run(text="The destruction of the rainforest.", args=args)
run(text="Trip", args=args)
run(text="Colonialism", args=args)
run(text="LSD", args=args)
run(text="Picasso", args=args)
run(text="Da vinci", args=args)
run(text="Mona Lisa", args=args)
run(text="Salvador Dali", args=args)
run(text="Dali", args=args)

quit()


run(text="A painting of a strong woman.", args=args)
run(text="A psychedelic experience on LSD.", args=args)
run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=consciousness, args=args)
run(text="Depression.", args=args)
run(text="Schizophrenia.", args=args)

args["style"] = "../stylegan2-ada-pytorch/logos_17040.pkl"
run(img="../deepdaze/logo.png", args=args)
run(img="../deepdaze/Logo_full.png", args=args)
run(text="The logo of a company named AdaLab.", args=args)
run(text="The logo of an A.I. startup named AdaLab.", args=args)
run(text="The logo of an A.I. startup.", args=args)

run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

quit()

run(text="A painting of a beautiful sunset.", args=args)
run(text="A painting of a strong man.", args=args)
run(img="base_images/Autumn_1875_Frederic_Edwin_Church.jpg", args=args)
run(img="base_images/anton_climb.jpg", args=args)
run(img="base_images/hot-dog.jpg", args=args)
run(text="A painting of a beautiful lake.", args=args)

quit()

args["style"] = "../stylegan2-ada-pytorch/logos_17040.pkl"

#run(text="A rocket.", args=args)
#run(text="A logo of a diving school at the silver lake.", args=args)
#run(text="A neural network", args=args)
#run(text="An alien claw", args=args)
#run(text="A logo in the style of a neural network.", args=args)
#run(text="A logo in the style of an alien claw.", args=args)

#run(img="../deepdaze/logo.png", args=args)
#run(img="../deepdaze/Logo_full.png", args=args)
#run(text="The logo of a company named AdaLab.", args=args)
#run(text="The logo of an A.I. startup named AdaLab.", args=args)
#run(text="The logo of an A.I. startup.", args=args)

#run(text="A painting of a sunflower", args=args)
#run(text=wizard, args=args)
#run(text=lama, args=args)
#run(text=consciousness, args=args)

run(text="A beautiful logo.", args=args)
for s in range(0, 4):
    run(text="A logo of a dance school", args=args, seed=s)


args["noise_opt"] = False
run(text="A logo of a rocket.", args=args)
run(text="A logo of a diving school.", args=args)
run(text="Artificial Intelligence.", args=args)
run(text="A logo of an artificial intelligence start-up.", args=args)

args["lr"] = 0.1
run(text="A logo of a rocket.", args=args)
run(text="A logo of a diving school.", args=args)
run(text="Artificial Intelligence.", args=args)
run(text="A logo of an artificial intelligence start-up.", args=args)

args["lr"] = 0.05
run(text="A logo of a rocket.", args=args)
run(text="A logo of a diving school.", args=args)
run(text="Artificial Intelligence.", args=args)
run(text="A logo of an artificial intelligence start-up.", args=args)

quit()



args["style"] = "../lucid-sonic-dreams/faces (ffhq config-f).pkl"
#run(text="A painting of a rose.", args=args)

args["lr_schedule"] = 1
#run(text="A painting of a rose.", args=args)
#run(text="Consciousness.", args=args)


args["lr_schedule"] = 0
args["start_image_path"] = 'base_images/anton_climb.jpg'
run(text="A beautiful person.", args=args)
quit()
run(text="Consciousness.", args=args)


args["start_image_path"] = 'base_images/hot-dog.jpg'
run(text="A painting of a rose.", args=args)
run(text="Consciousness.", args=args)

quit()

args["style"] = "../lucid-sonic-dreams/faces (ffhq config-f).pkl"
run(text="An enlightened person.", args=args)
run(text="Enlightenment.", args=args)
run(text="A depressive person.", args=args)
run(text="Depression.", args=args)
run(text="A demon.", args=args)
run(text="A criminal.", args=args)
run(text="A terrorist.", args=args)
run(text="A rich person.", args=args)
run(text="A poor person.", args=args)
run(text="A loving person.", args=args)
run(text="A hateful person.", args=args)


quit()

args["style"] = "../stylegan2-ada-pytorch/logos_4240.pkl"

#run(img="../deepdaze/logo.png", args=args)
#run(img="../deepdaze/Logo_full.png", args=args)
run(text="The logo of a company named AdaLab.", args=args)
run(text="The logo of an A.I. startup named AdaLab.", args=args)
run(text="The logo of an A.I. startup.", args=args)

run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

quit()

run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

quit()
run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["style"] = "../lucid-sonic-dreams/faces (ffhq config-f).pkl"
run(text="A painting of a sunflower", args=args)
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

quit()

args["noise_opt"] = False
args["reg_noise"] = False
args["regularize_noise_weight"] = 1e4
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["noise_opt"] = True
args["reg_noise"] = False
args["regularize_noise_weight"] = 1e4
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["noise_opt"] = True
args["reg_noise"] = True
args["regularize_noise_weight"] = 1e4
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["noise_opt"] = True
args["reg_noise"] = True
args["regularize_noise_weight"] = 1e3
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["noise_opt"] = True
args["reg_noise"] = True
args["regularize_noise_weight"] = 1e5
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)


args["epochs"] = 3
args["noise_opt"] = False
args["reg_noise"] = False
args["regularize_noise_weight"] = 1e4
args["lr"] = 1e-4
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["lr"] = 1e-5
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["lr"] = 1e-6
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

args["lr"] = 1e-1
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)

quit()







args["avg_feats"] = True
args["averaging_weight"] = 0.2
run(text=wizard, args=args)
run(text=consciousness, args=args)
run(text="Depression.", args=args)

args["averaging_weight"] = 0.4
run(text=wizard, args=args)
run(text=consciousness, args=args)
run(text="Depression.", args=args)

args["averaging_weight"] = 0.6
run(text=wizard, args=args)
run(text=consciousness, args=args)
run(text="Depression.", args=args)

args["averaging_weight"] = 0.8
run(text=wizard, args=args)
run(text=consciousness, args=args)
run(text="Depression.", args=args)
args["averaging_weight"] = 1.0

args["hidden_size"] = 256
args["avg_feats"] = False
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 512
args["avg_feats"] = False
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 256
args["avg_feats"] = True
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 512
args["avg_feats"] = True
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

quit()

args["hidden_size"] = 64
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 128
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 256
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)

args["hidden_size"] = 512
run(text=wizard, args=args)
run(text=lama, args=args)
run(text=consciousness, args=args)
run(text="The sun setting spectaculously over the beautiful ocean.", args=args)
run(text="A painting of a sunset.", args=args)
run(text="A painting of a sunrise.", args=args)




quit()




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


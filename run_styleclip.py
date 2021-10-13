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


def run(text=None, img=None, encoding=None, name=None, audio=None, args=None, **kwargs):
    if args is None:
        args = {}
    args = copy.copy(args)
    for key in kwargs:
        args[key] = kwargs[key]

    img_name = img.split("/")[-1] if img is not None else None
    input_name = create_text_path(text=text, img=img_name, encoding=encoding, context_length=77, audio=audio)
    
    # switch to own folder
    original_dir = os.getcwd()
    time_str = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    model_type = args["model_type"]
    if model_type == "vqgan":
        model_type += str(args["codebook_size"])
    if model_type == "stylegan":
        remove = ["(", " ", ")", "/", "-", "[", "]"]
        style = args["style"].split("/")[-1].split(".")[0]
        for c in remove:
            style = style.replace(c, "")
        name = os.path.join("style_clip", str(style), time_str + "_" + input_name)
    else:
        name = os.path.join(model_type, time_str + "_" + input_name)
    name = os.path.join("generations", name)
    os.makedirs(name, exist_ok=True)
    # copy start image to folder
    args = dict(args)
    if "start_image_path" in args and (args["start_image_path"].endswith(".jpg") or args["start_image_path"].endswith(".png")):
        shutil.copy(args["start_image_path"], name)
    # copy image for feature extraction to folder
    #if img is not None and isinstance(img, str):
    #    img_new_name = img.split("/")[-1] # only take end path
    #    remove_list = [")", "(", "[", "]", '"', "'"]
    #    for char in remove_list:
    #        img_new_name = img_new_name.replace(char, "")
    #    shutil.copy(img, os.path.join(name, img_new_name))
    #    img = img_new_name

    try:        
        imagine = Imagine(text=text,
            img=img,
            clip_encoding=encoding,
            audio=audio,
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
        
        #imagine.set_clip_encoding(text=text,
        #    img=img,
        #    encoding=encoding,
        #    neg_text=args["neg_text"] if "neg_text" in args else None,
        #)
        
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
        plt.plot([l.cpu().item() for l in imagine.aug_losses], label="Augmented")
        plt.plot([l.cpu().item() for l in imagine.non_aug_losses], label="Raw")
        plt.legend()
        plt.savefig("loss_plot.png")
        plt.close()
       
        # save
        del imagine.perceptor
        del imagine.model.perceptor
        #torch.save(imagine.cpu(), "model.pt")
        del imagine

    finally:
        os.chdir(original_dir)
        
        
def create_encoding(text_list=None, img_list=None, text_weight=0.5, args=None, **kwargs):
    if text_list is None:
        text_weight = 0
    elif img_list is None:
        text_weight = 1
        
    if args is None:
        args = {}
    args = copy.copy(args)
    for key in kwargs:
        args[key] = kwargs[key]

    imagine = Imagine(
        save_progress=True,
        open_folder=False,
        save_video=True,
        **args
    )

    encode_imgs = torch.mean(torch.stack([imagine.create_img_encoding(img=i) for i in img_list]), dim=0) if img_list is not None else 0
    encode_texts = torch.mean(torch.stack([imagine.create_text_encoding(text=t) for t in text_list]), dim=0) if text_list is not None else 0

    return text_weight * encode_texts + (1 - text_weight) * encode_imgs


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
parser.add_argument("--codebook_size", default=1024, type=int)
parser.add_argument("--run", default=None)
parser.add_argument("--model_type", default="vqgan")
parser.add_argument("--num_layers", default=44, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--sideX", default=480, type=int)
parser.add_argument("--sideY", default=480, type=int)
parser.add_argument("--lr", default=0.1, type=float)

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
    texts = [text for text in texts if not text.startswith("#")]
    
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



#run(text="An alien", args=args, iterations=50)

#args["model_type"] = "vqgan"
args["save_every"] = 20
args["start_img_loss_weight"] = 0.0
args["start_image_steps"] = 500
#args["lr"] = 0.1
#args["sideX"] = 480
#args["sideY"] = 480
#args["batch_size"] = 32
args["iterations"] = 1500
# codebook_size = 1024, # [1024, 16384]




run_text = args.pop("run")
if run_text is not None and run_text != "":
    run(text=run_text, args=args)
    quit()


old_ones = ["A climber climbing a large mountain", "A monkey painting a painting", "An image of a dog having a spiritual experience", "Being born", "The process of dying", "Surfing a big wave in the ocean", "The all-seeing tree", "The blue all-seeing tree. A blue tree with a large eye in its stem", "The scream by Edvard Munch", "Demon", "Entering the gates of heaven", "Meeting God", "Satan", "The final judgement", "Demon", "The most beautiful painting", "The most ugly painting", "A beautiful person", "A criminal", "A cute person", "An ugly person", "A photo of a criminal", "A photo of a terrorist", "A poor person", "A rich person"]

#for p in old_ones:
#    run(text=p, args=args)

# mfn: 12 layers - lr=5e-4
# siren: 44 layers - lr=1e-5
#for p in old_ones:
#    run(text=p, args=args, sideX=256, sideY=256, batch_size=16, model_type="mfn", num_layers=12, hidden_size=256, lr=5e-4)

classics = ["A man painting a red painting.", "A wizard in blue robes is painting a red painting in a castle", "A psychedelic experience on LSD.", "Schizophrenia", "Depression", "Love"]
religion_gpt = ["Above, in its brilliance, a solitary sun spins once again, burned, to slow the pace of the shifting sea of Nature. Above, the gods sing as if they are from the infinite.", "The bark of another corpse bleeds with the blood of our bond, this one rotted and long forgotten. The entire God of the Sacrifice looks down upon our abode, a drowning world that is us.", "As you chant, so I weave my voice. I craft my strings of fire, I fill the world with my own sweet sound. The stars themselves, the ends of the land, the wind, even the place where the light goes out all bend to worship me.", "Our hearts are hard and our bodies are weak. Our mind is a temple of madness. To tear it down is to destroy the seeds of our own transformation. We cannot kill our darkness. It is within us, waiting to be liberated. The folly of the human spirit.", "The oceans are my delights, and as I sleep in them my children lie with me in dreams of pure light. Their mother is my constant and eternal sleep. Not of her own will, but by her own will. I have no desire to enter the lake of shadow. Not in the dusk, not when the shades bleed.", "The infinite regress of illusion is the shortest path to nothingness. The void is the core of the matter of illusion. The void is eternal and unmoving. All our movements are reflections of ourselves in a mirror.", "We are nothing more than a million dreamers entwined in an infinite dream, completely unaware of our own identities. When the tentacles of the dream begin to stir, we wake up.", "A thousand worlds died. The best of them will live on. But only in infinite circles. Forever and ever. Except we are not the best of them. We are not the creams-of-life of the understanding.", "Each of our mortal experiences is just one more system of realization of a theme determined by our inherited molecular-cellular code.", "The godheads who dare not study the law of the unknown whispered in my ears to put me in an architect-work of my own to predict the unknown. It was too much work for my ignorant eyes.", "Your vision is weakened by the path of sin, which will not leave you until you repent. Your hatred of God is deep. You fear to kneel before God, lest you be stained with his blood.", "It is easy to be sane in a castle surrounded by walls. It is easy to be sane in the company of friends. It is easy to be sane in the company with trappings of royalty. It is easy to be sane in the garden. To be sane in the company of gods is to be roundly miserable."]
bands_that_arent_real = ["Necrotic Tyrant - Lost Crypts of the Unremembered", "Apocalyptic Maelstrom - Darkwater Monoliths", "Of Ancient Times - Necropolis of the Blood Red Sun", "Sectomancer - Blood from a Dying Sun", "Beneath the Dark Sea - It Lurks in the Deep"]
prank_comics = ["Take a picture of your bathroom and plaster it on your fridge.", "Stand in line for a movie for 30 minutes then leave.", "Ordering junk treats from TV ads at 3 in the morning, like those 'falling in the ocean' doughnuts", "Eat a banan inside your own mouth", "Have a tortoise deliver your package for you, don't be surprised if it sings 'Happy Birthday' back at you.", "Opt for the treadmill. You'll be running in the nude.", "Put a cup of coffee on your lap. It's an oldie, but it's still a favourite.", "My cat slept through a December blizzard in Florida this year", "There's a square of chicken in your front yard", "Placing a crown on your head."]
pinar_1 = ["quantum physics", "God", "The soul of the world", "unconditional love", "Spiritual teacher", "Spirituality", "Goddess of the world", "Anima Mundi", "Twilight state", "Children of god", "mama matrix most mysterious", "psychic", "the power of magic", "psilocybin", "shamanism", "non-human intelligence", "alchemy", "Amongst the elves", "non-duality", "duality", "metaphysical sensitivity"]
fritz_imgs = ["base_images/fritzkola/wahrheit_einhorn.jpg", "base_images/fritzkola/wahrheit_fluch.jpg", "base_images/fritzkola/wahrheit_zombies.jpg", "base_images/fritzkola/wahrheiten_satan.jpg"]
fritz_imgs_cropped = ["base_images/fritzkola/wahrheit_einhorn_cropped.png", "base_images/fritzkola/wahrheit_fluch_cropped.png", "base_images/fritzkola/wahrheit_zombies_cropped.png", "base_images/fritzkola/wahrheit_satan_cropped.png"]
fritz_texts = ["Plastic bottles killed the last unicorn", "Each plastic bottle has a soul curse embedded in it", "Plastic bottles turn us into mindless zombies", "Plastic bottles come directly from Satan"]
fritz_texts_ger = ["Plastikflaschen töteten das letzte einhorn", "In jede plastikflasche ist ein seelenfluch eingearbeitet", "plastikflaschen machen uns zu willenlosen zombies", "Plastikflaschen kommen direkt von Satan"]
fritz_texts_desc = ["A dead unicorn, killed by plastic bottles.", "An evil witch holding a plastic bottle.", "Zombies with plastic bottles as brains. Zombies with heads filled with plastic bottles."]
fritz_satan = ["Plastic bottles come directly from Satan", "Plastic bottles are made by Satan", "Plastic bottles pooped by satan, plastic bottles come out of Satan's ass."]
neg_text = 'incoherent, confusing, cropped, watermarks'
neg_text_2 = 'incoherent, confusing, cropped, watermarks, anime'


def add_context(words, prefix="", suffix=""):
    return [prefix + word + suffix for word in words]

args["early_stopping_steps"] = 0
args["use_tv_loss"] = 1
args["neg_text"] = neg_text
args["iterations"] = 500

comics = add_context(fritz_texts, suffix=". Comic.")
comic_style = add_context(fritz_texts, suffix=". Comic-book style.")

satan_comics = add_context(fritz_satan, suffix=". Comic.")
satan_bw = add_context(fritz_satan, suffix=". A black and white illustration.")
satan_fritzkola_ad = add_context(fritz_satan, suffix=". In the style of a fritzkola advertisement.")
satan_fritzkola = add_context(fritz_satan, suffix=". Fritzkola.")


fritz_satan_subj = ["Satan creates plastic bottles", "Satan makes plastic bottles", "Satan poops plastic bottles", "Satan has plastic bottles coming out of his butt"]
fritz_satan_subj_bw = add_context(fritz_satan_subj, suffix=". A black and white illustration.")
fritz_satan_subj_comic = add_context(fritz_satan_subj, suffix=". Comic.")
fritz_alien = ["Plastic bottles are brought by alien invasors", "Plastic bottles come from alien invasors", "An alien shoots humans with a plastic bottle", "An alien zaps humans with a plastic bottle."]
fritz_alien_comic = add_context(fritz_alien, suffix=" .Comic.")
fritz_alien_desc = ["A laughing alien shoots a human, using a plastic bottle as a gun.", "An alien shoots a human, using a plastic bottle as a gun.", "An alien uses a plastic bottle as a gun to shoot a human.", "An evil, laughing alien with a plastic bottle.", "An evil, laughing alien is standing next to a plastic bottle"]


args["iterations"] = 2000
args["save_every"] = 20
args["averaging_weight"] = 0

args["decay_cutout"] = 0



pride_prompts = ["Pride month", "Pride", "Be who you are!", "Be who you are. Trending on artstation", "A football stadium colored in rainbow colors", "A football stadium colored in rainbow colors in support for the LGBTQ+ community", "People dancing in rainbows", "A painting of People dancing in rainbows", "People dancing in rainbows. Trending on artstation.", "Fight for your rights to express yourself!", "Fight for your rights to express yourself! A painting.", "Fight for your rights to express yourself! Trending on artstation"]
    
ai_art_prompts = ["Using artificial intelligence to create art.", "Using artificial intelligence to create art. A painting", "Using artificial intelligence to create art. Trending on artstation", "Using artificial intelligence to create art. Trending on reddit.", "Artificial intelligence is creative", "Using artificial intelligence as an artistic instrument", "Using artificial intelligence as an artistic instrument. A painting", "Using artificial intelligence as an artistic instrument. Trending on artstation"]
lilli_geschenk = ["A voucher for a vibrator", "A voucher for a dildo", "A voucher for a vibrating dildo", "A voucher for a colourful dildo"]
lilli_geschenk_2 = ["A gift card for a dildo", "A gift card for a dildo", "A dildo voucher", "A vibrator voucher", "A dildo gift card", "A 40€ voucher for a dildo", "A 40€ gift card for a dildo", "A gift card for a dildo, gifted by students"]

#late_night_prompts = [#'A beautiful logo that reads "Encourage Consulting"', 'A beautiful logo that has "Encourage Consulting" written on it', "Encourage. Consulting. Logo.", 
#late_night_prompts = ["A logo of a psychological personal coach.", "An amazing logo of a psychological personal coach.", "Psychological personal coaching", 
late_night_prompts = ["Female Psychological personal coaching", "Being coached by an expert psychologist", "Being born", "The process of dying", "The reaper harvests souls, one after the other", "Non-duality", "Death. Trending on artstation", "The Reaper. Trending on artstation", "A beautiful woman named Anna is playing spike ball", "Anna is playing spike ball", "Anna is playing spike-ball", "Anna is playing spike ball with her grandmother", "Anna and her grandmother are playing spike ball", "Anna and her grandmother are playing spike ball at the beach"]
sayings = ["Life isn’t about waiting for the storm to pass. It’s about dancing in the rain.", " A blessing in disguise", "Give someone the benefit of the doubt", "Let someone off the hook", "No pain, no gain", "No pain, no gain. Impressionist painting", "No pain, no gain. Expressionist painting", " Speak of the devil", "The best of both worlds", "A bird in the hand is worth two in the bush", "A perfect storm", "Costs an arm and a leg", "Comparing apples to oranges", "Every cloud has a silver lining", "Get a taste of your own medicine", " It takes one to know one", "Play devil's advocate", "Spill the beans", "Take it with a grain of salt", "The devil is in the details", "You can't judge a book by its cover", "Don't beat a dead horse", " He who laughs last laughs loudest", "Make hay while the sun shines", "Once bitten, twice shy", "When it rains it pours", "You can lead a horse to water, but you can't make him drink", "You can catch more flies with honey than you can with vinegar"]
ak_custom_prompts = ["Vector quantized generative adversarial network", "An alligator character with a cutoff shirt posing in a fron double biceps. In the style of a Disney cartoon.", "Octopus attacking a hard of mules in the style of Wayne Gretzky", "colorless green ideas sleep furiously", "Sick half pipe 360 in the style of Beksinksi", "Robot reading deep lerning papers", "Research paper which solves General artificial intelligence", "Dog wearing pants", "I look at myself in the mirror", "Artstation. Trendin gon Unreal engine", "Starbucks on the moon surrounded by nebulas"]

#for p in ["A blonde girl and her grandmother are playing with a spiky ball at the beach", "The soul leaves the body and transcends dimensions", "The fourth dimension", "Procrastination", "Uhhhm yeah.", "I have no idea", "What is this???", ":-)", ":-()", ":D", "The singularity"]:
#    run(text=p, args=args)

#for p in ["My dad with a new smartphone!", "How do I use this smartphone?!", "Wow, a new smartphone for my birthday???", "A birthday smartphone"]:
#    run(text=p, args=args)


strong_adjectives = ["Oneness", "Unity", "Non-duality", "Pureness", "purity", "independence", "freedom", "life", "the meaning of life", "Mind-bending", "Shocking", "Awe-inspiring", "Unbelievable", "Mind-boggling", "Exhilarating", "Electrifying", "Mesmerizing", "Earth-shattering", "Weird", "Catastrophical", "Apocalyptical", "Soul-shattering", "Tempostuous", "passionate", "torrid", "soulful", "ardent", "impassioned", "hysterical"]

cool_prompts = ["Looking back", "Supernova", "Boundless ego", "Oceanic boundlessness", "Consciousness", "Black hole", "Shifting", "The end of times", "a painting of a witch brewing a Halloween potion by Greg Rutkowski", "A green frog wearing a tiny hat", "Control the soul", "a landscape resembling The Lovers tarot card by Greg Rutkowski", "a beautiful epic wondrous fantasy painting of wind", "a beautiful epic wondrous fantasy painting of fire"]

football_imgs = ["When it rains, it pours", "A perfect storm", "Don't beat a dead horse", "You can't judge a book by its cover", "Spill the beans"]
nlu_impossible = ["tree", "Tree is a perennial plant, a biological type, with an elongated stem, or trunk, with supporting branches and leaves.", "The trophy did not fit in the suitcase because it was too big", "The trophy did not fit in the suitcase because it was too small", "Hey Joe, the loud omelet wants another beer", "The white house rebuked the threatening statements north korea made.", "I like to play bridge", "Mary enjoyed the movie", "Mary enjoyed the sandwich", "BBC has a reporter in every country", "John had pizza with his kids", "John had a pizza with pineapple", "The corner table wants another beer", "Don't worry about Simon, he's a rock", "John works in the neighbourhood store", "John works in the computer store"]


def multi(prompt_list, *args, **kwargs):
    for p in prompt_list:
        run(text=p, *args, **kwargs)
        
# FLORKA
args["rand_convex"] = False
args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["lr"] = 0.03
args["batch_size"] = 4
args["iterations"] = 2000


# prompts
dnd_land = ["A goblin", "A goblin. DnD style", "A goblin. Dungeons-and-dragons style", "A goblin. Epic creature", "Vampire lord", "Count Strahd von Zarovich is a vampire lord, who rules over the valley of Barovia. Ages ago he made a pact with the Dark Powers of the Shadowfell, granting him immortality. However, it turned him into a vampire, and the valley became his prison from which he can never escape.", "Tiamat", "The Dragon Goddess of Greed, Queen of Evil Dragons, all time enemy of Bahamut and servant of Asmodeus, the great Tiamat. She has a head of each main chromatic dragon type, which is able to cooperate, and fight together. She is a hateful and greedy dragon. It’s safe to say Tiamat is a foe to be reckoned with in battle.", "Orcus", "Orcus is the Master of the Undead. He does not care about anybody, not even his servants. His intentions are to spread evil, death and misery only. He is in possession of powerful artifacts, and weapons that make him even more dangerous."]
animal_land = ["A cute dog", "A cute dog", "A cute puppy with large eyes"]
future_land = ["An epic utopian futuristic city, integrated in nature", "An illustration of people sleeping in cryogenic pods.", "People sleeping in cryogenic pods"]
beach_marriage_land = ["An orca marriage at the beach", "Two orcas marrying at the beach"]
couch_cuddle_land = ["A beautiful couch where a couple can cuddle and relax", "The most comfortable cuddle area"]

root_folder = "base_images/florka_bilder/"
if os.path.exists(root_folder):
    florka_imgs = [root_folder + img for img in os.listdir(root_folder) if img.endswith(".jpeg") or img.endswith(".png")]
#print(florka_imgs)

dnd_2 = ["Dungeons and Dragons", "A dragon", "A portal to a different dimension", "A drow", "A tiefling", "A dragonborn"]
animal_2 = ["A mass of puppies with large eyes", "Cute puppies", "The cutest image I have ever seen!", "This animal is so cute!", "Baby apes", "Baby dolphins", "Baby kittens", "Kittens"]
future_2 = ["A rocket", "Elon Musk", "A beautiful city", "A skyline", "A city embedded in nature", "Spaceships", "rockets", "Laser guns", "Sci-fi", "Cyberpunk", "Neon cyberpunk", "Cyberpunk Utopia", "Life on mars", "A mars base", "A human base on mars", "Mars"]
beach_marriage_2 = ["A beautiful marriage on a small island in a lighthouse", "A beautiful marriage on the beach next to a lighthouse", "Beach marriage next to the lighthouse", "A beach marriage. Caspar David Friedrich."]
cuddle_2 = ["A couple cuddling in the most comfortable couch ever!", "This is so comfortable!", "Super comfy!"]

cool_sayings = ["The whole is more than the sum of its parts", "Together, forever"]
funny_marriage_sayings = ["When a man opens a car door for his wife, it’s either a new car or a new wife.", "My most brilliant achievement was my ability to be able to persuade my wife to marry me.", "If I get married, I want to be very married.", "By all means marry; if you get a good wife, you’ll become happy; if you get a bad one, you’ll become a philosopher.", "An archaeologist is the best husband a woman can have. The older she gets, the more interested he is in her.", "Marriage lets you annoy one special person for the rest of your life.", "Marriage is not just spiritual communion, it is also remembering to take out the trash."]
love_marriage_sayings = ["A successful marriage requires falling in love many times, always with the same person.", "A great marriage is not when the ‘perfect couple’ comes together. It is when an imperfect couple learns to enjoy their differences.", "Never marry the one you can live with, marry the one you cannot live without.", "A good marriage is one which allows for change and growth in the individuals and in the way they express their love.", "Marriage is not about age; it’s about finding the right person.", "Experts on romance say for a happy marriage there has to be more than a passionate love. For a lasting union, they insist, there must be a genuine liking for each other. Which, in my book, is a good definition for friendship."]

own_sayings = ["The whole is greater than the sum of its parts", "Everlasting love", "Cheesy love", "Cheesy"]

makeai_art = ["A painting of a monkey getting a haircut", "a microscopic German Shepherd dog", "olympic monkey wrestling for team GB", "a monkey swimming in the olympics"]


mama_tango_teacher = ["Psychologisch gesund führen mit weiblicher Energie!", "Leading psychologically healthy with female energy!", "Female effective & psychologically healthy leadership style", "From quota woman to quota hit", "From quota woman to quota hit with Tango technology", "From quota woman to quota hit with Tango technique"]

cake = ["A uniborn cake", "A uniborn", "A unicorn cake", "A cake in the shape of a unicorn", "A cake in a unicorn theme", "A cake made of unicorn flesh", "Wow Mum! This cake looks like a unicorn!", "Just give me a tasty and coherent-looking unicorn cake please goddammit..."]

deepdaze_prompts = ["mist over green hills", "shattered plates on the grass", "cosmic love and attention", "time traveler in the crowd", "life during the plague", "meditative peace in a sunlit forest", "a man painting a completely red image", "a psychedelic experience on LSD"]


nice_landscape= "A watercolor landscape with the sun over mountains covered in trees"

human_right_declaration_first_ten = ["All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.", 
                           "Everyone is entitled to all the rights and freedoms set forth in this Declaration, without distinction of any kind, such as race, colour, sex, language, religion, political or other opinion, national or social origin, property, birth or other status. Furthermore, no distinction shall be made on the basis of the political, jurisdictional or international status of the country or territory to which a person belongs, whether it be independent, trust, non-self-governing or under any other limitation of sovereignty.",
                           "Everyone has the right to life, liberty and security of person.", 
                           "No one shall be held in slavery or servitude; slavery and the slave trade shall be prohibited in all their forms.", 
                          "No one shall be subjected to torture or to cruel, inhuman or degrading treatment or punishment.",
                          "Everyone has the right to recognition everywhere as a person before the law.",
                          "All are equal before the law and are entitled without any discrimination to equal protection of the law. All are entitled to equal protection against any discrimination in violation of this Declaration and against any incitement to such discrimination.",
                          "Everyone has the right to an effective remedy by the competent national tribunals for acts violating the fundamental rights granted him by the constitution or by law.",
                          "No one shall be subjected to arbitrary arrest, detention or exile.",
                           "Everyone is entitled in full equality to a fair and public hearing by an independent and impartial tribunal, in the determination of his rights and obligations and of any criminal charge against him.",
                           
                          ]

deep_racers = ["Deep Racers", "DeepRacers", "Deep Racing Team", "Deep Learning", "Cool deep racers"]

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["iterations"] = 1000
args["codebook_size"] = 1024

args["batch_size"] = 8

run(text="intermezzo", args=args)

args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["batch_size"] = 16
args["gradient_accumulate_every"] = 4
multi(cool_prompts, args=args, start_image_path="white") 

args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["batch_size"] = 16
multi(cool_prompts, args=args)

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 32
multi(cool_prompts, args=args)


quit()

#run(text="speed test", clip_names=["ViT-B/16", "ViT-B/32", "RN50x16"], args=args) # 2.66it/s - 12 GB
#run(text="speed test", clip_names=["ViT-B/16"], args=args) # 4.5it/s
#run(text="speed test", clip_names=["ViT-B/32"], args=args) # 4.9it/s
#run(text="speed test", clip_names=["ViT-B/16", "ViT-B/32"], args=args) # 4.0it/s
#run(text="speed test", clip_names=["ViT-B/16", "RN50"], args=args) # 4.14it/s
#run(text="speed test", clip_names=["ViT-B/16", "RN50x16"], args=args) # 2.8it/s
#run(text="speed test", clip_names=["ViT-B/16", "ViT-B/32", "RN50"], args=args) # 3.77it/s



args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50x16"]
args["batch_size"] = 6
#multi(cool_prompts, args=args, start_image_path="white") 
# 2.7it/s

args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["batch_size"] = 16
#multi(cool_prompts, args=args, start_image_path="white") 
# 2.7it/s - 10.2GB

args["clip_names"] = ["ViT-B/16", "RN50"]
args["batch_size"] = 32
#multi(cool_prompts, args=args, start_image_path="white") 

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 8
#multi(cool_prompts, args=args, start_image_path="white") 
# 3.6it/s - 8.2GB

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 32
#multi(cool_prompts, args=args, start_image_path="white") 
# 2.2it/s - 10.9GB

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 32
args["gradient_accumulate_every"] = 4
#multi(cool_prompts, args=args, start_image_path="white") 
# 1.8s/it 

quit()


args["clip_names"] = ["ViT-B/16", "RN50"]
args["batch_size"] = 32
multi(cool_prompts, args=args, start_image_path="white")


args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 32
multi(cool_prompts, args=args)

# Vit and RN50x with res 480 and bs 8 need a bit under 12GB
args["clip_names"] = ["ViT-B/16", "RN50x16"]
args["batch_size"] = 8
args["gradient_accumulate_every"] = 4
multi(cool_prompts, args=args, start_image_path="white")

quit()

#multi(add_context(human_right_declaration_first_ten, prefix="A painting. ")
#, model_type="image", sideX=1024, sideY=1024, lr=0.05, batch_size=32, stack_size=8)

#multi(add_context(human_right_declaration_first_ten, prefix="A charcoal drawing. ")
#, model_type="image", sideX=1024, sideY=1024, lr=0.05, batch_size=32, stack_size=8)

#multi(human_right_declaration_first_ten, model_type="conv", sideX=1024, sideY=1024, lr=0.05, batch_size=32, stack_size=8, stride=1, downsample=True, num_channels=32, act_func="gelu", norm_type="layer", num_layers=5)

#multi(human_right_declaration_first_ten, model_type="siren", sideX=256, sideY=256, lr=1e-5, num_layers=32, hidden_size=256, batch_size=16)


#multi(human_right_declaration_first_ten[:3], model_type="image", sideX=1024, sideY=1024, lr=0.05, batch_size=64, stack_size=8, args=args)

#multi(human_right_declaration_first_ten[:3], model_type="conv", sideX=1024, sideY=1024, lr=0.05, batch_size=16, stack_size=8, stride=1, downsample=True, num_channels=32, act_func="gelu", norm_type="layer", num_layers=5, args=args)
#multi(human_right_declaration_first_ten, model_type="siren", sideX=256, sideY=256, lr=1e-5, num_layers=32, hidden_size=256, #batch_size=16, args=args)

#multi(deep_racers,args=args, model_type="vqgan", sideX=480, sideY=480, lr=0.1, batch_size=32)

#quit()
args["iterations"] = 2000

args["sideX"] = 640
args["sideY"] = 480
args["model_type"] = "vqgan"
args["lr"] = 0.1
args["batch_size"] = 12

run(text="A beautiful skyline of san diego next to a highway", img="base_images/michi_san_diego.jpg", args=args)
run(text="A beautiful skyline of los angeles next to a highway", img="base_images/michi_san_diego.jpg", args=args)

run(text="A beautiful skyline of san diego next to a highway", start_image_path="base_images/michi_san_diego.jpg", args=args)
run(text="A beautiful skyline of los angeles next to a highway", start_image_path="base_images/michi_san_diego.jpg", args=args)


run(img="base_images/michi_san_diego.jpg", args=args, start_image_path= "white")

#run(img="base_images/michi_san_diego.jpg", args=args)

#run(text="A green foggy city", img="base_images/michi_san_diego.jpg", args=args)
#run(text="A beautiful and foggy city", img="base_images/michi_san_diego.jpg", args=args)


#multi(add_context(human_right_declaration_first_ten, prefix="A marvellous painting about: "), model_type="vqgan", sideX=480, sideY=480, lr=0.1, batch_size=32, args=args)
#multi(add_context(human_right_declaration_first_ten, prefix="A marvellous painting about: "), model_type="vqgan", sideX=720, sideY=480, lr=0.1, batch_size=8, args=args)


#multi(add_context(human_right_declaration_first_ten, prefix="A fantastic painting about: "), model_type="vqgan", sideX=480, sideY=480, lr=0.1, batch_size=16, args=args)
args["neg_text"] = "text. watermarks. signature. scribbles"
#multi(add_context(human_right_declaration_first_ten, prefix="A fantastic painting about: "), model_type="vqgan", sideX=480, sideY=480, lr=0.1, batch_size=16, args=args)



quit()

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["model_type"] = "image"
args["sideX"] = 512
args["sideY"] = 512
args["iterations"] = 500
args["lr"] = 0.005
args["batch_size"] = 32
args["stack_size"] = 1
#multi(deepdaze_prompts, args=args)
args["stack_size"] = 2
#multi(deepdaze_prompts, args=args)
args["stack_size"] = 4
#multi(deepdaze_prompts, args=args)
args["stack_size"] = 6
#multi(deepdaze_prompts, args=args)

args["model_type"] = "image"
args["sideX"] = 1024
args["sideY"] = 1024
args["stack_size"] = 8
#multi(deepdaze_prompts, args=args)
#run(text=nice_landscape, args=args)

args["model_type"] = "conv"
args["lr_schedule"] = 1
args["lr"] = 0.005
args["num_channels"] = 3
args["num_layers"] = 5
args["batch_size"] = 32
args["act_func"] = "gelu"
args["downsample"] = True
args["norm_type"] = "layer"
args["stride"] = 1
#run(text=nice_landscape, args=args)


args["stack_size"] = 6
#multi(deepdaze_prompts, args=args)

args["model_type"] = "siren"
args["lr"] = 1e-5
args["num_layers"] = 8
args["hidden_size"] = 256
args["mini_batch_size"] = 0
args["sideX"] = 256
args["sideY"] = 256


args["stack_size"] = 1
#run(text=nice_landscape, args=args)

args["stack_size"] = 6
#run(text=nice_landscape, args=args)

args["stack_size"] = 1
args["num_layers"] = 44
args["hidden_size"] = 256
args["batch_size"] = 4
#run(text=nice_landscape, args=args)


args["model_type"] = "siren"
args["lr"] = 1e-5
args["num_layers"] = 44
args["hidden_size"] = 256
args["mini_batch_size"] = 0
args["sideX"] = 256
args["sideY"] = 256

args["iterations"] = 500
args["rand_convex"] = False

args["clip_names"] = ["ViT-B/32"]
args["batch_size"] = 16
args["gradient_accumulate_every"] = 4
#multi(deepdaze_prompts, args=args)


args["iterations"] = 2000
args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 8
args["num_layers"] = 32
multi(strong_adjectives, args=args)
multi(cool_prompts, args=args)


args["model_type"] = "image"
args["sideX"] = 1024
args["sideY"] = 1024
args["stack_size"] = 7
args["lr"] = 0.005
args["batch_size"] = 64
multi(strong_adjectives, args=args)
multi(cool_prompts, args=args)

quit()
args["clip_names"] = ["ViT-B/32"]
args["batch_size"] = 4
multi(deepdaze_prompts, args=args)

args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 4
args["gradient_accumulate_every"] = 4
multi(deepdaze_prompts, args=args)

quit()
args["rand_convex"] = False
args["clip_names"] = ["ViT-B/32"]
args["batch_size"] = 16
multi(deepdaze_prompts, args=args)

args["batch_size"] = 4
multi(deepdaze_prompts, args=args)

quit()
args["rand_convex"] = False
args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 4
multi(deepdaze_prompts, args=args)

#multi(strong_adjectives, args=args)
#multi(cool_prompts, args=args)


quit()



multi(cake, args=args)


quit()

# didnt work yet!
args["start_image_path"] = "white"
args["start_img_loss_weight"] = 0.0
multi(own_sayings, args=args)

quit()

multi(mama_tango_teacher, args=args)
multi(own_sayings, args=args)
multi(add_context(mama_tango_teacher, suffix=" A logo."), args=args)
multi(add_context(mama_tango_teacher, suffix=" A logo. Velvia."), args=args)
multi(add_context(mama_tango_teacher, suffix=" A logo. Velvet"), args=args)
multi(add_context(mama_tango_teacher, suffix=" A dynamic logo."), args=args)
multi(add_context(mama_tango_teacher, suffix=" A painting."), args=args)
multi(add_context(own_sayings, suffix=" A painting."), args=args)


#run(text="Don't worry about simon, he's a rock.", args=args)
#multi(nlu_impossible, args=args, iterations=1000)
#multi(cool_prompts, args=args)
#multi(funny_marriage_sayings, args=args)
#multi(love_marriage_sayings, args=args)
#multi(add_context(funny_marriage_sayings, prefix="A painting of the saying: "), args=args)
#multi(add_context(love_marriage_sayings, suffix=" A matte painting."), args=args)



#multi(makeai_art, args=args)

#multi(add_context(future_2, prefix="A futuristic illustration of "), args=args)





args["codebook_size"] = 8192 * 2

  

#multi(add_context(dnd_2, prefix="A painting of "), args=args)
#multi(add_context(animal_2, prefix="A painting of "), args=args,)
#multi(add_context(future_2, prefix="A futuristic illustration of "), args=args)
#multi(add_context(beach_marriage_2, prefix="A painting of "), args=args)
#multi(add_context(cuddle_2, prefix="A painting of "), args=args)


quit()

args["codebook_size"] = 8192
args["iterations"] = 100
args["sideX"] = 448
args["sideY"] = 448
args["save_every"] = 5
args["lr"] = args["lr"] * 0.5
args["start_img_loss_weight"] = 0.5
args["neg_text"] = None
cheek = florka_imgs[2]

#run(start_image_path=cheek, text="Pure fire, hell", args=args)
#run(start_image_path=cheek, text="A couple at a marriage", args=args)

ja = florka_imgs[5]
#run(start_image_path=ja, text="A happy couple marrying with balloons saying 'JA'", args=args, img=ja)

beach = florka_imgs[-1]

#run(start_image_path=beach, text="Beach party!", args=args)



#run(start_image_path=beach, img=beach, args=args)

#run(start_image_path=beach, img=beach, text="A lovely day at the beach", args=args)

#run(img=beach, text="A lovely day at the beach", args=args)

#run(start_image_path=beach, text="A lovely day at the beach", args=args)


#run(start_image_path=beach, img=beach, minus_text="", args=args)



#run(start_image_path=florka_imgs[2], text="A man kissing a woman on the cheek. They are frozen in ice", img=florka_imgs[2], args=args)

#run(start_image_path=florka_imgs[2], text="A man kissing a woman on the cheek. They are frozen in ice", args=args, start_img_loss_weight=2)


    
#for i in florka_imgs[:2]:
#    print(i)
#    run(start_image_path=i, text="A cute couple in love", args=args, lr=args["lr"] * 0.5, #start_img_loss_weight=0.5)
    
#for i in florka_imgs[:2]:
#    run(img=i, args=args)

quit()

# three models
multi(add_context(dnd_land, prefix="A painting of "), args=args)
# two models
multi(add_context(dnd_land, prefix="A painting of "), clip_names=["ViT-B/16", "ViT-B/32"], batch_size=8, args=args)
#multi(dnd_land, clip_names=["ViT-B/32"], batch_size=8, args=args)
#multi(dnd_land, clip_names=["ViT-B/16"], batch_size=8, args=args)

# rest
#multi(animal_land, args=args)
multi(add_context(future_land, prefix="A painting of "), args=args)
multi(add_context(beach_marriage_land, prefix="A painting of "), args=args)
multi(add_context(couch_cuddle_land, prefix="A painting of "), args=args)


quit()

args["iterations"] = 1000

#  test neutral and minus text in comparison to neg text

#for p in football_imgs:
#    run(text=p, args=args)

args["neg_text"] = None
#for p in football_imgs:
#    run(text=p, args=args)

args["minus_text"] = "A cropped image"
args["neutral_text"] = "An image"
#for p in football_imgs[1:]:
#    run(text=p, args=args)
    
    
args["minus_text"] = "A cropped, confusing, incoherent, watermarked image"
#for p in football_imgs[3:]:
#    run(text=p, args=args)

args["neutral_text"] = None
#for p in football_imgs:
#    run(text=p, args=args)
    
neg_text_3 = 'incoherent, confusing, cropped, watermarks, anime-style, cartoon-style, football-player, jersey'
args["neg_text"] = neg_text_3
args["minus_text"] = None
#for p in football_imgs:
#    run(text=p, args=args)

# metaphoical sayings
args["neg_text"] = neg_text
#for p in ["The metaphorical meaning of: " + s for s in sayings]:
#    run(text=p, args=args)
    
    
# test some strong adjectives and asking for the "meaning" of words/adjectives:
args["rand_convex"] = False
args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["lr"] = 0.03
args["batch_size"] = 4

#for p in ["The first dimension", "The second dimension", "The third dimension", "The fourth dimension", "The fifth dimension", "The soul leaves the body and transcends dimensions", "The soul leaves the body", "A painting of the soul leaving the body", "Transcending dimensions", "The meaning of life", "A painting displaying the meaning of life", "A shaman", "A painting of a shaman", "Behemoth", "Leviathan", "Polaris", "Vell-os", "A goblin", "A painting of a vampire", "A portrait of a vampire", "A titan", "Unrelentless power", "Pure destruction", "The source of the universe", "The source of all energy", "A ball made of pure energy", "A portal to a different dimension", "Anton", "Anton Wiehe", "Anna", "Anna Methner", "AdaLab", "A startup named 'AdaLab'"]:
#    run(text=p, args=args)

#for p in strong_adjectives:
#    run(text=p, args=args)
    
#for p in ["This painting is: " + adjective for adjective in strong_adjectives]:
#    run(text=p, args=args)
    
#for p in ["This painting represents the meaning of: " + adjective for adjective in strong_adjectives]:
#    run(text=p, args=args)

#for p in ["This hyperrealistic rendering represents the meaning of: " + adjective for #adjective in strong_adjectives]:
#    run(text=p, args=args)


#for p in ["This colourful illustration represents the meaning of of: " + adjective for adjective in strong_adjectives]:
#    run(text=p, args=args)


args["neg_text"] = None
#for p in ["This painting represents the meaning of: " + adjective for adjective in strong_adjectives]:
#    run(text=p, args=args)
    
#run(text=neg_text, args=args)
#run(minus_text=neg_text, args=args)

args["lr"] = 0.1
args["clip_names"] = ["ViT-B/16"]
args["batch_size"] = 8
#run(text=neg_text, args=args)
#run(minus_text=neg_text, args=args)
args["clip_names"] = ["ViT-B/32"]
#run(text=neg_text, args=args)
#run(minus_text=neg_text, args=args)

    
args["neg_text"] = neg_text
 
    

    
#for p in ["A 40€ voucher for a dildo", "Florka", "Florian and Sirka", "Florian and Sirka = Florka", "A painting of an orca marrying", "A painting of an orca marriage", "An orca marriage", "Two orcas marrying"]:
#    run(text=p, args=args)
    
#for p in ["How do I use this smartphone?!", "Wow, a new smartphone for my birthday???", "A birthday smartphone", "A smartphone on which you can play the game of Go"]:
#    run(text=p, args=args)

#for p in religion_gpt:
#    run(text=p, args=args)

quit()

args["iterations"] = 200
args["batch_size"] = 16

args["clip_names"] = ["ViT-B/16"]
#for p in classics:
#    run(text=p, args=args, vqgan_start_img_path="white")

#for p in ["Consciousness"]:
#    run(text=p, args=args)
#for p in sayings:
#    run(text=p, args=args)

args["clip_names"] = ["ViT-B/32"]

#for p in ["Consciousness"]:
#    run(text=p, args=args)
#for p in ["test"]:
#    run(text=p, args=args)

args["batch_size"] = 8
args["clip_names"] = ["ViT-B/16", "ViT-B/32"]

#for p in ["Consciousness"]:
#    run(text=p, args=args)
#for p in sayings:
#    run(text=p, args=args)

#for p in classics:
#    run(text=p, args=args)

args["rand_convex"] = False

args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["lr"] = 0.03
args["batch_size"] = 4
args["iterations"] = 2000
#for p in classics:
#    run(text=p, args=args)
    
args["clip_names"] = ["ViT-B/16", "ViT-B/32"]
args["batch_size"] = 8
#for p in classics:
#    run(text=p, args=args)

    

args["rand_convex"] = True
args["clip_names"] = ["ViT-B/16", "ViT-B/32", "RN50"]
args["lr"] = 0.03
args["batch_size"] = 4
#for p in ak_custom_prompts:
#    run(text=p, args=args)
    
    
args["lr"] = 0.1
args["clip_names"] = ["ViT-B/32"]
args["batch_size"] = 32
#for p in ak_custom_prompts:
#    run(text=p, args=args)
    
quit()

    

args["sideX"] = 256  # audioclip model is larger, 256 is nearly max size
args["sideY"] = 256
args["model_name"] = "audioclip"

args["audio_len"] = 1
args["neg_text"] = None
args["batch_size"] = 8

base_imgs = [os.path.join("base_images", p) for p in os.listdir("base_images") if p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".png")]
for p in base_imgs:
    if os.path.isdir(p):
        for subfile in os.listdir(p):
            base_imgs.append(os.path.join(p, subfile))
        base_imgs.remove(p)
audio_prompts = ["A dog barking", "A cat meowing", "A man screaming", "A woman screaming", "Classical music", "Techno", "Guitar sounds", "Beautiful music"]

args["model_type"] = "lstm_audio"
args["lr"] = 0.0002
args["audio_len"] = 5
#run(text="Classical music", args=args)

quit()

args["model_type"] = "siren_audio"
args["num_layers"] = 32
args["hidden_size"] = 256
args["lr"] = 5e-4
#run(text="A dog barking", args=args)

args["audio_channels"] = 2
#run(text="A dog barking", args=args)


quit()

args["iterations"] = 500
args["model_type"] = "vqgan"
args["model_name"] = "ViT-B/32"
#run(text="A dog barking", args=args)


args["model_type"] = "stylegan"
args["opt_raw"] = False
args["lr"] = 0.01
#run(text="A dog barking", args=args)

args["opt_raw"] = True
args["lr"] = 0.005
#run(text="A dog barking", args=args)


args["model_name"] = "ViT-B/32"
args["model_type"] = "mfn"
args["num_layers"] = 10
args["hidden_size"] = 256
#args["lr"] = 5e-4  # works nicely
#run(text="A dog barking", args=args)
#args["lr"] = 5e-3
#run(text="A dog barking", args=args)

args["model_type"] = "siren"
#args["lr"] = 5e-3
#run(text="A dog barking", args=args)
#args["lr"] = 5e-4
#run(text="A dog barking", args=args)
#args["lr"] = 5e-5 # gives results...
#run(text="A dog barking", args=args)
args["lr"] = 1e-5
run(text="A dog barking", args=args)
args["lr"] = 5e-6
run(text="A dog barking", args=args)
quit()

args["iterations"] = 2000
args["model_type"] = "unagan_audio"
args["lr"] = 1e-7#0.00001
args["unagan_type"] = "singing"
args["batch_size"] = 64
run(text="A dog barking", args=args)

quit()

args["iterations"] = 2000
args["model_type"] = "wavegrad_audio"
args["lr"] = 0.0001
run(text="A dog barking", args=args)

quit()



args["iterations"] = 2000
args["model_type"] = "conv_audio"
args["lr"] = 0.005 # 0.005 for stride==1
args["num_channels"] = 8
args["num_layers"] = 5
args["batch_size"] = 8
args["act_func"] = "relu"
args["downsample"] = False
args["norm_type"] = None
args["stride"] = 1
#run(text="A dog barking", args=args)
#for p in audio_prompts:
#    run(text=p, args=args)
#for p in base_imgs:
#    run(img=p, args=args)

args["model_type"] = "raw_audio"
args["lr"] = 0.005
#for p in audio_prompts:
#    run(text=p, args=args)
#for p in base_imgs:
#    run(img=p, args=args)



args["model_type"] = "lstm_audio"
args["lr"] = 0.0005
#for p in audio_prompts:
#    run(text=p, args=args)
#for p in base_imgs:
#    run(img=p, args=args)
    


args["model_type"] = "siren_audio"
args["num_layers"] = 32
args["hidden_size"] = 256
args["lr"] = 5e-6
# for imgs: lr=1e-5, num_layers=44, hidden_size=256
for p in audio_prompts:
    run(text=p, args=args)
for p in base_imgs:
    run(img=p, args=args)
    
   
    
quit()
    
args["model_type"] = "raw_audio"
args["lr"] = 0.005
run(text="A dog barking", args=args)


#gnossienne_img = "vqgan1024/2021-06-29_21:16:11_your_encoding/your_encoding.jpg"
#nadia_wuff_img = "vqgan1024/2021-06-29_22:13:33_nadia_wuff_/nadia_wuff_.jpg"
#run(img=gnossienne_img, args=args)

#run(text="A man screaming", args=args)

quit()
run(text="A man screaming", args=args)

quit()

import os


song_path = "base_audios"
songs = os.listdir(song_path)
songs = [s for s in songs if s.endswith(".mp3") or s.endswith(".wav")]
for s in songs:
    print(s)
    run(text=None, audio=os.path.join(song_path, s), args=args)

quit()
    
lyric_song_path = "../lucid-sonic-dreams/songs_with_lyrics"
songs = os.listdir(lyric_song_path)
songs = [s for s in songs if s.endswith(".mp3")]

#for s in songs:
#    print(s)
#    run(text=None, audio=os.path.join(lyric_song_path, s), args=args)
    
    
song_path = "../lucid-sonic-dreams/songs"
songs = os.listdir(song_path)
songs = [s for s in songs if s.endswith(".mp3")]
#for s in songs:
#    print(s)
#    run(text=None, audio=os.path.join(song_path, s), args=args)


#run(text=None, audio="base_audios/nadia_wuff.wav", args=args)
#run(text=None, audio="base_audios/me_speaking_test.wav", args=args)


quit()

four_k = ["landscape 4k", "forest 4k", "sea 4k", "blood 4k", "beauty 4k", "boobs 4k"]
for k in four_k:
    run(text=k, args=args)

quit()

for p in lilli_geschenk_2:
    run(text=p, args=args)

    
args["latent_type"] = "code_sampling"
for p in lilli_geschenk_2:
    run(text=p, args=args)
    
quit()

args["codebook_size"] = 8192
args["sideX"] = 440  # 8192 model is a bit bigger, so 440 instead of 480
args["sideY"] = 440

args["lr"] = 0.05  # 2 times lower
#for p in classics:
#    run(text=p, args=args)

# maybe we need a way lower learning rate for this model...
args["lr"] = 0.01  # 10 times lower

#for p in classics:
#    run(text=p, args=args)
    
args["lr"] = 0.005  # 20 times lower
#for p in classics:
#    run(text=p, args=args)
    
# maybet this model works better with sampling!
args["lr"] = 0.1  # 20 times lower
args["latent_type"] = "code_sampling"
#for p in classics:
#    run(text=p, args=args)

    
args["latent_type"] = "code_sampling"
args["lr"] = 0.01  # 10 times lower
#for p in classics:
#    run(text=p, args=args)

quit()

#run(text="A time-traveller in the crowd", args=args)
#run(text="Life during the plague", args=args)

# might need this
#pip install omegaconf>=2.0.0 pytorch-lightning>=1.0.8 einops>=0.3.0


run(text="A bisexual in the crowd", args=args)
run(text="A queer person in the crowd", args=args)
run(text="Colour in the crowd", args=args)
run(text="A rainbow flag swaying in the wind", args=args)
run(text="Pride month", args=args)
run(text="Pride. Trending on artstation", args=args)
run(text="Pride month. Trending on artstation", args=args)
run(text="Gradient", args=args)
run(text="Pure Madness", args=args)
run(text="Madness", args=args)
run(text="The smallest object in the universe", args=args)
run(text="The largest object int he universe", args=args)
run(text="The smallest object", args=args)
run(text="The smallest object that exists", args=args)
run(text="The largest object", args=args)
run(text="THe largest object that exists", args=args)

run(text="God, sitting on a cloud, is observing humans", args=args)
run(text="The Devil himself", args=args)
run(text="Life is meaningless", args=args)

run(text="A roboter is painting using watercolors", args=args)
run(text="An artist is being replaced by A.I.", args=args)
run(text="An artist is replaced by a robot", args=args)
run(text="A robot replaces an artist", args=args)
run(text="A robot is more creative than an artist", args=args)
run(text="A robot", args=args)
run(text="An android", args=args)


args["codebook_size"] = 8192
args["sideX"] = 440  # 8192 model is a bit bigger, so 440 instead of 480
args["sideY"] = 440

#for p in classics:
#    run(text=p, args=args)

args["codebook_size"] = 1024
for p in classics:
    run(text=p, args=args)

args["codebook_size"] = 16384
for p in classics:
    run(text=p, args=args)

quit()

run_from_file("poems/best_poems.txt", args=args, create_story=1, iterations=500, save_every=3)

quit()


for prompt in pride_prompts:
    run(text=prompt, args=args)

for prompt in ai_art_prompts:
    run(text=prompt, args=args)

    
quit()

run(text="Beksinski", args=args)
run(text="visions of the future inside a crystal ball", args=args)
run(text=' Fantasy forest painting called "Saga Flame" by Seb McKinnon, trending on Artstation.', args=args)
run(text="Meditative peace in a sunlit forest.", args=args)
run(text="Meditative peace in a sunlit forest. Trending on artstation.", args=args)
quit()



for p in fritz_alien:
    run(text=p, args=args)

for p in fritz_alien_desc:
    run(text=p, args=args)
    
quit()


for p in fritz_alien:
    run(text=p, args=args, model_type="conv", num_layers=2, stride=2, act_func="gelu")
    run(text=p, args=args, model_type="siren", lr=1e-5, num_layers=44, hidden_size=256, mini_batch_size =0, sideX=256, sideY=256)


quit()

for p in add_context(fritz_alien, suffix=" A movie poster."):
    run(text=p, args=args)

for p in add_context(fritz_alien, suffix=" An album cover."):
    run(text=p, args=args)

for p in add_context(fritz_alien, suffix=" An illustration."):
    run(text=p, args=args)
for p in add_context(fritz_alien, suffix=" An illustration of an alien."):
    run(text=p, args=args)
for p in add_context(fritz_alien, suffix=" In the style of Picasso."):
    run(text=p, args=args)
for p in add_context(fritz_alien, suffix=" A painting."):
    run(text=p, args=args)


quit()

for prompt in fritz_alien:
    run(text=prompt, args=args)

for prompt in fritz_alien_comic:
    run(text=prompt, args=args)

#for prompt in fritz_satan_subj:
#    run(text=prompt, args=args)
    
#for prompt in fritz_satan_subj_bw:
#    run(text=prompt, args=args)
    
#for prompt in fritz_satan_subj_comic:
#    run(text=prompt, args=args)
    
args["model_type"] = "conv"
args["sideX"] = 480
args["sideY"] = 480
args["iterations"] = 1000
args["lr_schedule"] = 1
args["lr"] = 0.005 * (args["sideX"] * args["sideY"] / 480 / 480)
args["num_channels"] = 3
args["num_layers"] = 2
args["batch_size"] = 32
args["act_func"] = "gelu"
args["downsample"] = True
args["norm"] = "layer"


def run_all(*args, **kwargs):
    run(*args, **kwargs, model_type="conv", lr=0.005, num_channels=3, num_layers=2, act_func="gelu", downsample=True, norm="layer")

    run(*args, **kwargs, model_type="siren", lr=1e-5)

    run(*args, **kwargs, model_type="vqgan")

    run(*args, **kwargs, model_type="stylegan")

    run(*args, **kwargs, args=args, model_type="siren", lr=5e-5)

args["iterations"] = 2000
args["sideX"] = 1920
args["sideY"] = 1080

args["lr"] = 0.005 * (args["sideX"] * args["sideY"] / 480 / 480)

run(text="A man painting a red painting. Single layer", args=args, model_type="conv", num_layers=1, stride=2, act_func="gelu", iterations=200)

run(text="A man painting a red painting. Two layers", args=args, model_type="conv", num_layers=2, stride=2, act_func="gelu", iterations=200)
for prompt in religion_gpt:
    run(text=prompt, args=args, act_func="gelu", num_layers=1, stride=2, downsample=False)
    
for prompt in religion_gpt:
    run(text=prompt, args=args, act_func="gelu", num_layers=1, stride=2, downsample=True)


run(text="A man painting a red painting", args=args, model_type="siren", lr=1e-5, num_layers=44, hidden_size=256, mini_batch_size = 2 ** 16, sideX=720, sideY=576, iterations=200)
for prompt in religion_gpt:
    run(text=prompt, args=args, model_type="siren", lr=1e-5, num_layers=44, hidden_size=256, mini_batch_size = 2 ** 16, sideX=720, sideY=576)

    
quit()
    
run(text="A painting of a vampire", args=args, num_layers=5, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a vampire", args=args, num_layers=5, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params

for prompt in classics:
    run(text=prompt, num_layers=5, stride=1, act_func=None, args=args)

for prompt in classics:
    run(text=prompt, num_layers=5, stride=1, act_func="gelu", args=args)

run(text="A painting of a vampire", args=args, num_layers=10, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a vampire", args=args, num_layers=10, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params


run(text="A painting of a vampire", args=args, num_layers=10, num_channels=32, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a vampire", args=args, num_layers=10, num_channels=32, stride=1, act_func="relu") # 4.6 it/s 691000 params
    
quit()
    
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params

args["num_channels"] = 64
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params
    
quit()
    
run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params

run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=2, act_func=None) # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=2, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=2, downsample=True, stride=2, act_func="relu") # 4.6 it/s 691000 params
    
quit()
    
run(text="A painting of a ghoul", args=args, num_layers=5, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=5, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params
run(text="A painting of a ghoul", args=args, num_layers=5, downsample=True, stride=1, act_func="relu") # 4.6 it/s 691000 params

quit()
    
    
run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params

run(text="A painting of a monster", args=args, num_layers=5, downsample=True, stride=1, act_func="gelu") # 4.6 it/s 691000 params

quit()
    
run(text="A painting of a vampire", args=args, num_layers=3, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params

    
quit()
    
run(text="A painting of a vampire", args=args, num_layers=1, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params

quit()
    
run(text="Speed test - one layer, stride 1 - conv - no act func", args=args, num_layers=1, downsample=True, stride=1, act_func=None) # 4.6 it/s 691000 params

quit()
    
run(text="Speed test - one layer, stride 1 - transpose conv", args=args, num_layers=1, downsample=False, stride=1) # 4.76 it/s 691281 2.7GB

run(text="Speed test - one layer, stride 1 - conv", args=args, num_layers=1, downsample=True, stride=1) # 4.6 it/s 691000 params

    
quit()
    
run(text="Speed test - one layer - upsample new zero init", args=args, num_layers=1, downsample=False) # 4.6 it/s 172000 params

    
quit()
    
run(text="Speed test - one layer - upsample", args=args, num_layers=1, downsample=False) # 4.6 it/s
run(text="Speed test - two layers - upsample", args=args, num_layers=2, downsample=False) # 4.3 it/s
run(text="Speed test - four layers - upsample", args=args, num_layers=4, downsample=False) # 4.4 it/s - 3300 params

    
quit()
    
#run(img="base_images/carmen_cute.png", args=args, model_type="siren", lr=1e-4)
#run(img="base_images/carmen_cute.png", args=args, model_type="siren", lr=5e-4)
#run(img="base_images/carmen_cute.png", args=args, model_type="siren", lr=1e-3)
run(text="Speed test - one layer", args=args, num_layers=1) # 3.9it/s
run(text="Speed test - two layers", args=args, num_layers=2) # 2.01 it/s


quit()

run(img="base_images/carmen_cute.png", args=args, num_layers=3) # 1.4s/it

run(img="base_images/carmen_cute.png", args=args, num_channels=32) # 1.75it/s
run(img="base_images/carmen_cute.png", args=args, num_channels=32, num_layers=3)
run(img="base_images/carmen_cute.png", args=args, num_channels=128)
run(img="base_images/carmen_cute.png", args=args, num_channels=256)



quit()

run(img="base_images/carmen_cute.png", args=args, lr=0.0025)
run(img="base_images/carmen_cute.png", args=args, lr=0.001)
run(img="base_images/carmen_cute.png", args=args, lr=0.0005)
run(img="base_images/carmen_cute.png", args=args, lr=0.0001)




quit()

run(img="base_images/carmen_cute.png", args=args, lr=0.01)
run(img="base_images/carmen_cute.png", args=args, lr=0.005)


quit()


quit()

run(img="base_images/carmen_cute.png", args=args)

run(img="base_images/carmen_cute.png", args=args, model_type="siren", lr=1e-5)

run(img="base_images/carmen_cute.png", args=args, model_type="vqgan")

run(img="base_images/carmen_cute.png", args=args, model_type="stylegan")




quit()


run(text="A painting of a vampire.", args=args, num_layers=2)
run(text="A painting of a ghost.", args=args, num_layers=2)
run(text="A painting of a monster.", args=args, num_layers=2)
run(text="A painting of a human.", args=args, num_layers=2)
for prompt in classics:
    run(text=prompt, args=args)




run(text="A painting of a vampire.", args=args, num_layers=2)
run(text="A painting of a vampire.", args=args, num_layers=3)

quit()

#run(text="A painting of a vampire.", args=args)
#run(text="A painting of a vampire.", args=args, num_layers=2)
#run(text="A painting of a vampire.", args=args, num_layers=3)

run(text="A painting of a vampire.", args=args, num_channels=8)
run(text="A painting of a vampire.", args=args, num_channels=16)
run(text="A painting of a vampire.", args=args, num_channels=64)




quit()

args["act_func"] = "relu"
run(text="A painting of a monster.", args=args, downsample=True)
run(text="A painting of a monster.", args=args)

args["act_func"] = "gelu"
run(text="A painting of a monster.", args=args, downsample=True)
run(text="A painting of a monster.", args=args)

quit()

run(text="A painting of a monster.", args=args, downsample=True)
run(text="A painting of a monster.", args=args)



quit()


run(text="A painting of a vampire.", args=args, noise_augment=False)
run(text="A painting of a vampire.", args=args, decay_cutout=1)

quit()


run(text="A painting of a vampire.", args=args, lr=.04)
run(text="A painting of a vampire.", args=args, lr=.0005)
run(text="A painting of a vampire.", args=args, lr=.00025)

quit()

run(text="A vampire. Unreal engine", args=args)
run(text="A painting of a vampire. Unreal engine.", args=args)

quit()
run(text="A painting of a vampire.", args=args, batch_size=1)
run(text="A painting of a vampire.", args=args, lr=.01)
run(text="A painting of a vampire.", args=args, lr=.02)

quit()


run(text="A painting of a fairy.", args=args, batch_size=4)
run(text="A painting of a fairy.", args=args, batch_size=8)
run(text="A painting of a fairy.", args=args, batch_size=16)
run(text="A painting of a fairy.", args=args, batch_size=32)
run(text="A painting of a fairy.", args=args, lr=0.02)
run(text="A painting of a fairy.", args=args, lr=0.0075)




quit()

    
args["model_type"] = "siren"
args["sideX"] = 256
args["sideY"] = 256
args["num_layers"] = 44
args["hidden_size"] = 256
args["lr_schedule"] = 0
args["lr"] = 1e-5


for prompt in fritz_satan_subj:
    run(text=prompt, args=args)
    
for prompt in fritz_satan_subj_bw:
    run(text=prompt, args=args)
    
for prompt in fritz_satan_subj_comic:
    run(text=prompt, args=args)

quit()



args["sideX"] = 512
args["sideY"] = 512
args["iterations"] = 500
args["save_every"] = 20

args["model_type"] = "conv"
#run(text="A colourful bird drinking out of a beautiful flower", args=args, num_layers=1)
#run(text="A colourful bird drinking out of a beautiful flower", args=args, num_layers=4)
#run(text="A colourful bird drinking out of a beautiful flower", args=args, num_layers=8)

run(text="A colourful bird drinking out of a beautiful flower", args=args, num_channels=64)
run(text="A colourful bird drinking out of a beautiful flower", args=args, num_channels=1024)
run(text="A colourful bird drinking out of a beautiful flower", args=args, num_layers=4, num_channels=512)



quit()

args["sideX"] = 1920
args["sideY"] = 1080
args["iterations"] = 2000
args["save_every"] = 20

#args["model_type"] = "siren"
#for prompt in classics:
#    run(text=prompt, args=args)

args["model_type"] = "image"
run(text="A colourful bird drinking out of a beautiful flower", args=args)


quit()

for prompt in classics:
    run(text=prompt, args=args)

quit()

for prompt in classics:
    run(text=prompt, args=args)
for prompt in classics:
    run(text=prompt, args=args, decay_cutout=1)
for prompt in classics:
    run(text=prompt, args=args, averaging_weight=0)
for prompt in classics:
    run(text=prompt, args=args, averaging_weight=1)
for prompt in classics:
    run(text=prompt, args=args, averaging_weight=0, decay_cutout=1)
for prompt in classics:
    run(text=prompt, args=args, averaging_weight=1, decay_cutout=1)

    
    


quit()

args["iterations"] = 2000
args["decay_cutout"] = 0
for prompt in fritz_satan:
    run(text=prompt, args=args)
for prompt in satan_comics:
    run(text=prompt, args=args)
for prompt in satan_bw:
    run(text=prompt, args=args)
for prompt in satan_fritzkola_ad:
    run(text=prompt, args=args)
for prompt in satan_fritzkola:
    run(text=prompt, args=args)



quit()

#for prompt in comics:
#    run(text=prompt, args=args, iterations=2000)
#for prompt in comics:
#    run(text=prompt, args=args, decay_cutout=1, iterations=2000)  
#for prompt in comic_style:
#    run(text=prompt, args=args, iterations=2000)
#for prompt in comic_style:
#    run(text=prompt, args=args, decay_cutout=1, iterations=2000)

    
for img in fritz_imgs_cropped[1:]:
    run(img=img, args=args, iterations=2000)
    
    

    
args["iterations"] = 2000
args["decay_cutout"] = 1
for prompt in classics:
    run(text=prompt, args=args)
for prompt in religion_gpt:
    run(text=prompt, args=args)
args["decay_cutout"] = 0
for prompt in classics:
    run(text=prompt, args=args)
for prompt in religion_gpt:
    run(text=prompt, args=args)
    
quit()

for prompt in fritz_texts_desc:
    run(text=prompt, args=args)

for prompt in add_context(fritz_texts_desc, suffix=". A black and white illustration."):
    run(text=prompt, args=args)

for prompt in fritz_texts:
    run(text=prompt, args=args, decay_cutout=1)
    
for prompt, img in zip(fritz_texts, fritz_imgs_cropped):
    run(text=prompt, img=img, args=args)
    
for prompt, w in zip(fritz_texts, [0.9, 0.95, 0.99]):
    fritz_encoding = create_encoding(text_list=prompt, img_list=fritz_imgs_cropped, args=args, text_weight=w)
    run(encoding=fritz_encoding, args=args)
    
for prompt in fritz_texts:
    run(text=prompt, args=args, iterations=2000)
    
for prompt in fritz_texts:
    run(text=prompt, args=args, decay_cutout=1, iterations=2000)

quit()

for prompt in fritz_texts:
    fritz_encoding = create_encoding(text_list=prompt, img_list=fritz_imgs_cropped, args=args, text_weight=0.8)
    run(encoding=fritz_encoding, args=args)
    
for prompt in add_context(fritz_texts, suffix=". A black and white illustration."):
    run(text=prompt, args=args)

    
for prompt in fritz_texts:
    run(text=prompt, args=args, iterations=2000)
    
quit()

# pure img mean
#run(encoding=create_encoding(img_list=fritz_imgs, args=args), args=args)
#run(encoding=create_encoding(img_list=fritz_imgs_cropped, args=args), args=args)

# img mean + text
#for prompt in fritz_texts:
#    fritz_encoding = create_encoding(text_list=prompt, img_list=fritz_imgs, args=args, text_weight=0.5)
#    run(encoding=fritz_encoding, args=args)

for prompt in fritz_texts:
    fritz_encoding = create_encoding(text_list=prompt, img_list=fritz_imgs_cropped, args=args, text_weight=0.5)
    run(encoding=fritz_encoding, args=args)
    
# pure text
#for prompt in fritz_texts:
#    run(text=prompt, args=args)

for prompt in fritz_texts_ger:
    run(text=prompt, args=args)
    
for prompt in add_context(fritz_texts, suffix=". A black and white illustrated adverstisement."):
    run(text=prompt, args=args)


quit()

#run(text="David Bowie", args=args, neg_text=neg_text)

args["model_type"] = "siren"
args["sideX"] = 512
args["sideY"] = 512
args["num_layers"] = 44
args["hidden_size"] = 252
args["lr_schedule"] = 0
args["lr"] = 1e-5
#args["lr"] = 0.5


prompt = "internet"
args["mini_batch_size"] = 2** 13
print("Mini batch size: ", args["mini_batch_size"])

args["decay_cutout"] = 1
args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

quit()

run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)


quit()
run(text=prompt, args=args, neg_text=neg_text)


quit()

args["decay_cutout"] = 1
args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

quit()

args["neg_text"] = neg_text
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

args["neg_text"] = neg_text_2
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)

quit()

prompt = "Internet"

quit()

run(text="Tiamat", args=args, neg_text=neg_text_2)

quit()
run(text="Anime", args=args, neg_text=neg_text_2)
quit()

run(text=prompt, args=args, neg_text="AI-generated")
run(text=prompt, args=args, neg_text="blurry")
run(text=prompt, args=args, neg_text="ugly")



quit()
run(text=prompt, args=args, neg_text="artificially generated image")
run(text=prompt, args=args, neg_text="cropped, watermarks")
run(text=prompt, args=args, neg_text="anime")
run(text="tiamat", args=args, neg_text="anime")


quit()
prompt = "tiamat"
run(text=prompt, args=args)
run(text=prompt, args=args, neg_text="incoherent")
run(text=prompt, args=args, neg_text="confusing")
run(text=prompt, args=args, neg_text="cropped")
run(text=prompt, args=args, neg_text="watermarks")
run(text=prompt, args=args, neg_text=neg_text)

quit()

run(text="beautiful coral reef. RTX on", args=args)
run(text="beautiful coral reef. HD rendered. RTX.", args=args)
run(text="beautiful coral reef. RTX off", args=args)

args["neg_text"] = None
run(text=neg_text, args=args)

quit()

args["model_type"] = "siren"
args["sideX"] = 256
args["sideY"] = 256
args["num_layers"] = 32
args["hidden_size"] = 256
args["lr_schedule"] = 0
args["lr"] = 1e-5,
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)

quit()

args["use_tv_loss"] = 1
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["use_tv_loss"] = 0

args["decay_cutout"] = 1
run(text="H R Giger", args=args)
run(text="Rainforest", args=args)
run(text="Night club", args=args)
run(text="seascape painting", args=args)
run(text="Flowing water", args=args)
run(text="Internet", args=args)
run(text="Logo of an A.I. startup named AdaLab", args=args)
args["decay_cutout"] = 0


quit()

run(text="The forest wizard. Unreal engine", args=args, iterations=500, decay_cutout=1)
run(text="The forest wizard. Unreal engine", args=args, iterations=500, decay_cutout=0)

quit()
    
for prompt in pinar_1:
    run(text=prompt, args=args, circular=0)
    
    
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


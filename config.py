import torch
from util.common import L_RGB, L_DEPTH, L_NORMAL

DATASET = 'bdataset_tiny'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
BATCH_SIZE = 4
IMAGE_SIZE = 224
MAP_SIZE = 32
NUM_CLASSES = 30
ENCODER_LEARNING_RATE = 1e-3
DECODER_LEARNING_RATE = 1e-3
ENCODER_LR_MILESTONES = [150]
DECODER_LR_MILESTONES = [150]
BETAS = (.9, .999)
GAMMA = .5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 3
VOXEL_THRESH = [.2, .3, .4, .5]
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
AUGMENT = False
TEST = True
PLOT = False
OUT_PATH = "./"
CHECKPOINT_FILE = "tiny.pth"
IMG_DIR = "../" + DATASET + "/images/"
MESH_DIR = "../" + DATASET + "/labels/"
DETECT_PATH = "./data/images"
USED_LAYERS = [L_RGB, L_DEPTH, L_NORMAL]

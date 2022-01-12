import logging
from argparse import ArgumentParser

LATENT_SIZE = 64 # GAZEMAE 128 -> autoencoder / 2
RAND_SEED = 123

##
# training settings
MAX_TRAIN_ITERS = 20000
CHECKPOINT_INTERVAL = 500

# network settings
ENCODER_KERNEL_SIZE = 3
ENCODER_FILTERS = [256, 256, 256, 256]
ENCODER_DILATIONS = [(1, 1), (2, 4), (8, 16), (32, 64)]
ENCODER_DOWNSAMPLE = [0, 0, 0, 0]

#ENCODER_PARAMS = [ENCODER_KERNEL_SIZE, ENCODER_FILTERS, ENCODER_DILATIONS,
#                  ENCODER_DOWNSAMPLE]

DECODER_FILTERS = [128, 128, 128, 128]
DECODER_DILATIONS = [(1, 1), (2, 4), (8, 16), (32, 64)]
DECODER_KERNEL_SIZE = 3
DECODER_INPUT_DROPOUT = 0.66  # BEST = 0.66 for vel, 0.75 for pos
DECODER_PARAMS = [DECODER_KERNEL_SIZE, DECODER_FILTERS, DECODER_DILATIONS,
                  DECODER_INPUT_DROPOUT]

##new
DECODER_HZ = 0
DECODER_VIEWING_TIME = -1

# data settings
MAX_X_RESOLUTION = 1280
MAX_Y_RESOLUTION = 1024
DATA_ROOT = 'data/'
GENERATED_DATA_ROOT = 'generated-data/'

SLICE_OVERLAP_RATIO = 0.2

PX_PER_DVA = 35  # pixels per degree of visual angle

#######################################################################################################################
#Args, arguments that I want to change, depending on the round

# means the doesn’t need to go into the code and make changes to the script. 
# Giving the user the ability to enter command line arguments provides flexibility.
def get_parser():
    parser = ArgumentParser()
    #parser.add_argument("-l", "--log-to-file", default=False, action="store_true")
    #parser.add_argument("-v", "--verbose", default=False, action="store_true")
    parser.add_argument("--save-model", default=True, action="store_true")
    parser.add_argument("--tensorboard", default=False, action="store_true")
    # Data Settings
    parser.add_argument("-hz", default=0, type=int)
    parser.add_argument("-vt", "--viewing-time", help="Cut raw gaze samples to this value (seconds)", default=-1, type=float)
    parser.add_argument("--signal-type", default='pos', type=str, help="'pos' or 'vel'")
    parser.add_argument("--slice-time-windows", default=None, type=str, help="'2s-overlap' or '2s-disjoint'")
    parser.add_argument("--augment", default=False, action="store_true")#?
    # Training Settings
    parser.add_argument("--use-validation-set", default=False, action="store_true")
    parser.add_argument("-bs", "--batch-size", default=64, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", default=5e-4, type=float)
    parser.add_argument("--model-pos", default='', type=str)#name of a saved model: pos-i3738, used if we want to load it
    parser.add_argument("--model-vel", default='', type=str)#name of a saved model: vel-i8528

    return parser

def print_settings():
    logging.info({
        k: v for (k, v) in globals().items()
        if not k.startswith('_') and k.isupper()})


def setup_logging(args, run_identifier=''):
    logging.getLogger().setLevel(logging.INFO)
    if args.log_to_file:
        for handler in logging.root.handlers[:]:
            logging.roost.removeHandler(handler)
        log_filename = run_identifier + '.log'
        logging.basicConfig(filename='../logs/' + log_filename,
                            level=logging.INFO,
                            format='%(message)s')
        logging.getLogger().addHandler(logging.StreamHandler())
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s')
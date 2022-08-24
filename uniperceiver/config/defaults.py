# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 1

_C.NAME = '' # task name


# -----------------------------------------------------------------------------
# Shared targets
# -----------------------------------------------------------------------------
_C.SHARED_TARGETS = []
_C.SHARED_TARGETS_CFG = CN()
_C.SHARED_TARGETS_CFG.FILE_PATH = ''
_C.SHARED_TARGETS_CFG.DISTRIBUTED = False

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN() #

_C.DATASETS.TRAIN = ''

_C.DATASETS.VAL = ''

_C.DATASETS.TEST = ''

_C.DATASETS.TASK_TYPE = ''
_C.DATASETS.DATASET_NAME = ''
_C.DATASETS.TARGET_SET = ['']
_C.DATASETS.TRAIN_BATCH_SIZE = 64
_C.DATASETS.TEST_BATCH_SIZE = 64
_C.DATASETS.VERSION = 'v1'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

_C.DATALOADER.UNIFIED_DATASET = False

_C.DATALOADER.FAKE_DATA = False

_C.DATALOADER.TASK_TYPE = ''

_C.DATALOADER.TRAIN_BATCH_SIZE = 64

_C.DATALOADER.TEST_BATCH_SIZE = 64

_C.DATALOADER.NUM_WORKERS = 4

_C.DATALOADER.FEATS_FOLDER = ''

_C.DATALOADER.LOCAL_PREFIX=''

_C.DATALOADER.SAMPLER=''
_C.DATALOADER.CACHE_MODE=True

_C.DATALOADER.APPEND_EOS=True
_C.DATALOADER.ONE_STREAM=True
_C.DATALOADER.RANDOM_MASK=True




_C.DATALOADER.LOCAL_PREFIX=''
_C.DATALOADER.CLASS_NAME_FILE = ''

_C.DATALOADER.VISUAL_FEAT = True
_C.DATALOADER.ANNO_FOLDER = ''
_C.DATALOADER.ANNO_FILENAME = None
_C.DATALOADER.S3_PATH = ''
_C.DATALOADER.S3_ANNO_FOLDER = None
_C.DATALOADER.CIRCULAR_CACHE_MODE = False
_C.DATALOADER.ZIP_MODE = False
_C.DATALOADER.CACHE_ORIGIN_IMAGE = False
_C.DATALOADER.RANDOM_CAPTION = True
_C.DATALOADER.AS_NUMPY_AS_POSSIBLE = False

_C.DATALOADER.RELATION_FILE = ''

_C.DATALOADER.GV_FEAT_FILE = ''

_C.DATALOADER.ATTRIBUTE_FILE = ''

_C.DATALOADER.SEQ_PER_SAMPLE = 5
_C.DATALOADER.MIN_SEQ_PER_SAMPLE = 5

_C.DATALOADER.MAX_FEAT_NUM = -1

_C.DATALOADER.NEGATIVE_SIZE = -1

_C.DATALOADER.INF_BATCH_SIZE = 200 # for single stream retrieval only, chunk size

_C.DATALOADER.USE_GLOBAL_V = True

_C.DATALOADER.USE_WEIGHTED_SAMPLER = False

_C.DATALOADER.SAMPLING_WEIGHT = 1.0
_C.DATALOADER.TRANSFORM =  ''



# xiaoshi: added for video cls
_C.DATALOADER.FRAMES_PER_CLIP = 4
_C.DATALOADER.STRIDE = 5
_C.DATALOADER.FILE_EXTENSION = ''
_C.DATALOADER.ANNO_FILE = 'annotation.json'
_C.DATALOADER.TIMESFORMER_AUG = False

# hao:
_C.DATALOADER.DO_AS_RETRIEVAL = False

_C.DATALOADER.USE_CEPH = False

# xiaoshi: added for vqa, specify inference mode
_C.DATALOADER.DO_AS_GEN = True
_C.DATALOADER.VQA_INPUT = ['image', 'question']
_C.DATALOADER.SINGLE_CLASS = False
_C.DATALOADER.SMALL_VAL = True
_C.DATALOADER.BLOCK_VQ = False
_C.DATALOADER.DATA_PERCENTAGE = 1.0
_C.DATALOADER.TWO_EOT = False
_C.DATALOADER.DATA_K_SAMPLE = -1

_C.DATALOADER.PIN_MEM = True
_C.DATALOADER.PREFETCH_FACTOR = 2
_C.DATALOADER.PADDING_TO_MAX = False
_C.DATALOADER.LOAD_INLABEL = True



_C.DATALOADER.MULTI_VEIW_NUM = 1
_C.DATALOADER.MULTI_VEIW = 'v0'




_C.TASKS = [] # task config

_C.ENCODERS = [] # multi encoder config
# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------
_C.ENGINE = CN()

_C.ENGINE.NAME = ''

_C.ENGINE.MIXUP = 0.
_C.ENGINE.CUTMIX = 0.
_C.ENGINE.MIXUP_PROB = 0.
_C.ENGINE.MIXUP_SWITCH_PROB = 0.0
_C.ENGINE.MIXUP_MODE = ''
_C.ENGINE.MIXUP_LABEL_SMOOTHING = 0.0

# change to dataloader
_C.DATALOADER.MIXUP = 0.
_C.DATALOADER.CUTMIX = 0.
_C.DATALOADER.MIXUP_PROB = 0.
_C.DATALOADER.MIXUP_SWITCH_PROB = 0.0
_C.DATALOADER.MIXUP_MODE = ''
_C.DATALOADER.MIXUP_LABEL_SMOOTHING = 0.0

_C.DATALOADER.MINI_BATCHES = 1
_C.DATALOADER.SYNC_TASK = False
_C.DATALOADER.STRATEGY = ''
_C.DATALOADER.TURN_LOG = True
_C.DATALOADER.TCS_CONF_PATH = 'petreloss.config'
_C.DATALOADER.NUM_GTS = 1
_C.DATALOADER.USE_SEG_ID = False


# -----------------------------------------------------------------------------
# Scheduled sampling
# -----------------------------------------------------------------------------
_C.SCHEDULED_SAMPLING = CN()

_C.SCHEDULED_SAMPLING.START_EPOCH = 0

_C.SCHEDULED_SAMPLING.INC_EVERY_EPOCH = 5

_C.SCHEDULED_SAMPLING.INC_PROB = 0.05

_C.SCHEDULED_SAMPLING.MAX_PROB = 0.25

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()

_C.MODEL.DEVICE = "cuda"

_C.MODEL.TEMP_NAME = ""

_C.MODEL.IMG_INPUT_SIZE = 224
_C.MODEL.PATCH_SIZE = 16

_C.MODEL.BLOCK_IMAGENET = False

_C.MODEL.FAKE_PAD_TO_MAX = False

_C.MODEL.VOCAB_SIZE = 1000 # include <BOS>/<EOS>

_C.MODEL.META_ARCHITECTURE = ''

_C.MODEL.ENCODER = ''

_C.MODEL.ENCODER_DIM = 1024

_C.MODEL.DECODER = ''

_C.MODEL.DECODER_DIM = 1024

_C.MODEL.PRED_DROPOUT = 0.0

_C.MODEL.PREDICTOR = ''

_C.MODEL.V_PREDICTOR = ''

_C.MODEL.USE_PREDICTOR_BIAS = False
_C.MODEL.SHARE_PREDICTOR_HIDDEN = False
_C.MODEL.SHARE_CLS_NAME_QUERY_EMBED = False

_C.MODEL.PRED_TEMPERATURE = 1.0

_C.MODEL.PRED_USE_NORM = True

_C.MODEL.MAX_SEQ_LEN = 17

_C.MODEL.EVAL_MAX_SEQ_LEN = 17

_C.MODEL.MAX_LABEL_LEN = 5

_C.MODEL.WEIGHTS = ''

_C.MODEL.ITM_NEG_PROB = 0.5

# used for image patch
_C.MODEL.CLS_TOKEN = False
# xiaoshi: added for video cls
_C.MODEL.BACKBONE = 'deit_base'
_C.MODEL.CENTRAL_FRAME_INIT = False

_C.MODEL.SHARE_MODULES = []

_C.MODEL.PROMPT = False

_C.MODEL.PROMPT_PARAM = []
_C.MODEL.FC_PROMPT = False
_C.MODEL.FC_PROMPT_OUT = -1
_C.MODEL.TWO_LOSS = False
_C.MODEL.FC_BIAS = 0.0
_C.MODEL.FC_PROMPT_WEIGHTS = 'learn'
_C.MODEL.FC_PROMPT_INDEX = -1


_C.MODEL.GEN_MASK = True
_C.MODEL.SKIP_WORD_EMB = False
_C.MODEL.IN_TUNING = False

# ----------------------------------------------------------------------------
# Token embedding
# ----------------------------------------------------------------------------
_C.MODEL.TOKEN_EMBED = CN()

_C.MODEL.TOKEN_EMBED.NAME = ''

_C.MODEL.TOKEN_EMBED.DIM = 1024

_C.MODEL.TOKEN_EMBED.ACTIVATION = 'none'

_C.MODEL.TOKEN_EMBED.ELU_ALPHA = 0.5

_C.MODEL.TOKEN_EMBED.USE_NORM = False

_C.MODEL.TOKEN_EMBED.DROPOUT = 0.0

_C.MODEL.TOKEN_EMBED.POSITION = 'none'

_C.MODEL.TOKEN_EMBED.POSITION_MAX_LEN = 5000

_C.MODEL.TOKEN_EMBED.TYPE_VOCAB_SIZE = 0

_C.MODEL.TOKEN_EMBED.TYPE_SEG_SIZE = 0

_C.MODEL.OLD_CHECKPONT = True

# ----------------------------------------------------------------------------
# Visual embedding
# ----------------------------------------------------------------------------
_C.MODEL.VISUAL_EMBED = CN()

_C.MODEL.VISUAL_EMBED.NAME = ''

_C.MODEL.VISUAL_EMBED.IN_DIM = 2048

_C.MODEL.VISUAL_EMBED.OUT_DIM = 1024

_C.MODEL.VISUAL_EMBED.ACTIVATION = 'none'

_C.MODEL.VISUAL_EMBED.ELU_ALPHA = 0.5

_C.MODEL.VISUAL_EMBED.USE_NORM = False

_C.MODEL.VISUAL_EMBED.DROPOUT = 0.0

_C.MODEL.VISUAL_EMBED.LOCATION_SIZE = 0

_C.MODEL.VISUAL_EMBED.TYPE_SIZE = 0 # type embedding for image

_C.MODEL.VISUAL_EMBED.PATCH_SIZE = 16

_C.MODEL.VISUAL_EMBED.IMAGE_SIZE = 224

# video embedding

_C.MODEL.VIDEO_EMBED = CN()

_C.MODEL.VIDEO_EMBED.NAME = ''

_C.MODEL.VIDEO_EMBED.IN_DIM = 2048

_C.MODEL.VIDEO_EMBED.OUT_DIM = 1024

_C.MODEL.VIDEO_EMBED.ACTIVATION = 'none'

_C.MODEL.VIDEO_EMBED.ELU_ALPHA = 0.5

_C.MODEL.VIDEO_EMBED.USE_NORM = False

_C.MODEL.VIDEO_EMBED.DROPOUT = 0.0

_C.MODEL.VIDEO_EMBED.POSITION = 'none'

_C.MODEL.VIDEO_EMBED.MAX_LENGTH = 1000

_C.MODEL.VIDEO_EMBED.TYPE_SIZE = 0 # type embedding for image

_C.MODEL.VIDEO_EMBED.ADD_TYPE_EMBED = False

_C.MODEL.VIDEO_EMBED.PATCH_SIZE_S = 16

_C.MODEL.VIDEO_EMBED.PATCH_SIZE_T = 8

_C.MODEL.VIDEO_EMBED.DIVIDE_ST_POS = False

_C.MODEL.VIDEO_EMBED.USE_VISUAL_TOKENIZER = False

_C.MODEL.VIDEO_EMBED.USE_VISUAL_POS = False

_C.MODEL.VIDEO_EMBED.MAX_FRAMES = 8
_C.MODEL.VIDEO_EMBED.POS_RANDOM = True

# video tokenizer

_C.MODEL.VIDEO_TOKENIZER = CN()

# _C.MODEL.VIDEO_TOKENIZER.PATCH_SIZE_S = 16

# _C.MODEL.VIDEO_TOKENIZER.PATCH_SIZE_T = 8

_C.MODEL.VIDEO_TOKENIZER.FPS = -1 # -1 means using a fixed number of frames

_C.MODEL.VIDEO_TOKENIZER.NUM_FRAMES = 50 # works only when VIDEO_TOKENIZER.NUM_FRAMES == -1

_C.MODEL.VIDEO_TOKENIZER.SAMPLE_OFFSET = 0

_C.MODEL.VIDEO_TOKENIZER.MAX_FRAMES = 40


# xiaoshi: added for video cls
_C.MODEL.NUM_CLASSES = 339

#
_C.MODEL.PRETRAIN = False

_C.MODEL.FIX_PRETRAIN_PARAM = True

_C.MODEL.USE_ORIGINAL_CODER = False


# prompt embedding

_C.MODEL.PROMPT_EMBED = CN()

_C.MODEL.PROMPT_EMBED.NAME = "none"

_C.MODEL.PROMPT_EMBED.DIM = 512

_C.MODEL.PROMPT_EMBED.PROMPT_LENGTH = 10
_C.MODEL.PROMPT_EMBED.TARGET_PROMPT_LENGTH = 10
_C.MODEL.PROMPT_EMBED.INPUT_DEEP_PROMPT_LENGTH = 10
_C.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT_LENGTH = 10


_C.MODEL.PROMPT_EMBED.ACTIVATION = 'none'

_C.MODEL.PROMPT_EMBED.ELU_ALPHA = 0.5

_C.MODEL.PROMPT_EMBED.USE_NORM = False

_C.MODEL.PROMPT_EMBED.DROPOUT = 0.0

_C.MODEL.PROMPT_EMBED.WITH_POS = False

_C.MODEL.PROMPT_EMBED.INPUT_PROMPT = False
_C.MODEL.PROMPT_EMBED.TARGET_PROMPT = False

_C.MODEL.PROMPT_EMBED.DEEP_PROMPT = False
_C.MODEL.PROMPT_EMBED.TARGET_DEEP_PROMPT = False
_C.MODEL.PROMPT_EMBED.SHARE_DEEP_PROMPT = False

_C.MODEL.PROMPT_EMBED.LABLE_PROMPT = False
_C.MODEL.PROMPT_EMBED.LABEL_SIZE = 0

# ----------------------------------------------------------------------------
# Pre-training
# ----------------------------------------------------------------------------
_C.MODEL.PRETRAINING = CN()

_C.MODEL.PRETRAINING.MODEL_NAME = 'bert-base-uncased'

_C.MODEL.PRETRAINING.FROM_PRETRAINED = 'bert-base-uncased'

_C.MODEL.PRETRAINING.DO_LOWER_CASE = True

# ----------------------------------------------------------------------------
# BERT
# ----------------------------------------------------------------------------
_C.MODEL.BERT = CN()

_C.MODEL.BERT.SCALE_MULTI_BEFORE = False

_C.MODEL.BERT.DROP_PATH_PROB = 0.0

_C.MODEL.BERT.DROP_PATH_PROB_FIXED = False


_C.MODEL.BERT.HIDDEN_SIZE = 512

_C.MODEL.BERT.HIDDEN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.HIDDEN_ACT = "gelu"

_C.MODEL.BERT.NUM_ATTENTION_HEADS = 8

_C.MODEL.BERT.INTERMEDIATE_SIZE = 2048

_C.MODEL.BERT.INTERMEDIATE_DROP = 0.1

_C.MODEL.BERT.FFN_DROPOUT_PROB = 0.1

_C.MODEL.BERT.ATTENTION_PROBS_DROPOUT_PROB = 0.1

_C.MODEL.BERT.V_TARGET_SIZE = 0

_C.MODEL.BERT.NUM_HIDDEN_LAYERS = 12

_C.MODEL.BERT.LAYER_DROP = 0.0

_C.MODEL.BERT.V_NUM_HIDDEN_LAYERS = 6

_C.MODEL.BERT.V_LAYER_DROP = 0.0

_C.MODEL.BERT.NUM_UNDERSTANDING_LAYERS = 6

_C.MODEL.BERT.U_LAYER_DROP = 0.0

_C.MODEL.BERT.NUM_GENERATION_LAYERS = 6

_C.MODEL.BERT.G_LAYER_DROP = 0.0

_C.MODEL.BERT.SKIP_TARGET_ENCODE = False
_C.MODEL.BERT.NORMALIZE_BEFORE = False
_C.MODEL.BERT.NORMALIZE_DECISION = ''
_C.MODEL.BERT.QKV_BIAS = True

_C.MODEL.BERT.UNIFY_QKV = True

_C.MODEL.FEATURE_GATHER = False
_C.MODEL.FEATURE_GATHER_FORCE = False

_C.MODEL.LEARN_TEMP = False

_C.MODEL.LABELS_NUM = 1000
_C.MODEL.TRANSFORM = True

_C.MODEL.QUEUE_LEN = 1024

_C.MODEL.SwitchParamsInit = False
_C.MODEL.TimmParamsInit = False
_C.MODEL.MAEParamsInit = False
_C.MODEL.MOCOv3ParamsInit = False
_C.MODEL.POSEMBEDFIX = False
_C.MODEL.POSEMBED_SCALE = 1.0
_C.MODEL.CHECKPOINT_FILETER = True
_C.MODEL.CHECKPOINT_FILETER_VIDEO = True
_C.MODEL.TimmParamsInitSTD = 0.02
_C.MODEL.TimmParamsINIT_EMBEDDING_STD = 0.02

_C.MODEL.SHARE_LAYERNORM = False
_C.MODEL.BW_WORD_ALONE = False

_C.MODEL.BW_EMBED_SPE = True
_C.MODEL.WORD_SEPERATE = True
_C.MODEL.BW_OWD_EMBED = False
_C.MODEL.TEXT_VISUAL_SEPARATE = False
_C.MODEL.OUTPUT_PROJ = False
_C.MODEL.POS_BEFORE = True
_C.MODEL.LN_FP32 = False
_C.MODEL.GATE_FP32 = False
_C.MODEL.TAG_TRANSFORM_FP32 = False
_C.MODEL.MODEL_EMA = False
_C.MODEL.MODEL_EMA_DECAY = 0.9999
_C.MODEL.MODEL_EMA_FORCE_CPU = False

_C.MODEL.LAYER_SCALE = False
_C.MODEL.LAYER_SCALE_INIT = 1e-5
_C.MODEL.LAYER_SCALE_FP32 = True

_C.MODEL.MASK_RAND = False
_C.MODEL.MASK_RATIO = 0.25
_C.MODEL.MIXUP_ALIGN = False

_C.MODEL.LAYER_TOKEN_MASK = False
_C.MODEL.LAYER_MASK_IDX = [4]
_C.MODEL.LAYER_MASK_RATIO = [0.25]

_C.MODEL.TOKEN_EMBED_COPY = False
_C.MODEL.TOKEN_EMBED_VALID_END = 128


# ----------------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------------
_C.SOLVER = CN()

_C.SOLVER.NAME = 'Adam'
_C.SOLVER.DEEPSPEED = True

_C.SOLVER.RESUME_OPTIMIZER = False

_C.SOLVER.TORCH_OPTIMIZER = False
_C.SOLVER.PARAMS_SEPERATE = False
_C.SOLVER.PARAMS_GROUP = False

_C.SOLVER.TORCH_OPTIMIZER = False
_C.SOLVER.PARAMS_SEPERATE = False
_C.SOLVER.PARAMS_GROUP = False

_C.SOLVER.EPOCH = 10

_C.SOLVER.MAX_ITER = 10000

_C.SOLVER.CHECKPOINT_PERIOD = 1

_C.SOLVER.CHECKPOINT_MAX_SAVE = 1000

_C.SOLVER.EVAL_PERIOD = 1

_C.SOLVER.BASE_LR = 0.0005

_C.SOLVER.ACCUM_ITER = 0

_C.SOLVER.BIAS_LR_FACTOR = 1.0

_C.SOLVER.WG_LR_FACTOR = 1.0

_C.SOLVER.LR_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY = 0.0

_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.WEIGHT_DECAY_NORMBIAS_WEIGHT = True

_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.WEIGHT_DECAY_WG = 0.0

_C.SOLVER.WEIGHT_DECAY_EMBEDDING = 0.05

_C.SOLVER.OUTPUTPROJ_NOWD = False

_C.SOLVER.INITIAL_ACCUMULATOR_VALUE = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.DAMPENING = 0.0

_C.SOLVER.NESTEROV = 0.0

_C.SOLVER.ALPHA = 0.99

_C.SOLVER.BETAS = [0.9, 0.999]

_C.SOLVER.EPS = 1e-8

_C.SOLVER.AMSGRAD = False

_C.SOLVER.CENTERED = False

_C.SOLVER.GRAD_CLIP_TYPE = 'norm' # norm, value

_C.SOLVER.GRAD_CLIP = 0.1

_C.SOLVER.MIN_LOSS_SCLE = 2048.0
_C.SOLVER.LOSS_SCALE_WINDOW = 500

_C.SOLVER.NORM_TYPE = 2.0

_C.SOLVER.WRITE_PERIOD = 20
_C.SOLVER.GradHistogram = False
_C.SOLVER.GradHistogramPeriod  = 200

_C.SOLVER.COMPUTE_MOE_DECISION  = False


_C.SOLVER.LOG_GRAD  = False
_C.SOLVER.LOG_GRAD_ITER  = 300


_C.SOLVER.AMP_FP16 = False

_C.SOLVER.APEX_FP16 = False

_C.SOLVER.APEX_OPT_LEVEL = 'O1'
_C.SOLVER.APEX_MASTER_WEIGHTS = True

_C.SOLVER.FUSED_LAYERNORM = False

_C.SOLVER.BF16 = False

_C.SOLVER.ZEROSTAGE = 0

# used by xiaoshi in default trainer
_C.SOLVER.FP16 = False


_C.SOLVER.GRAD_PRINT = False

_C.SOLVER.CHECKPOINT_MAPPING = []
_C.SOLVER.CHECKPOINT_MAP = True
_C.SOLVER.RESUME_TAU = True
_C.SOLVER.CHECKPOINT_CEPH_SAVE = False

_C.SOLVER.BALANCE_LOSSESS = False
_C.SOLVER.BALANCE_LOSSESS_WEIGHT = 0.01
_C.SOLVER.CONSISTENCE_LOSSESS = 0.01
_C.SOLVER.DIVEGENCE_LOSSESS = 0.01
_C.SOLVER.WORD_BALANCE_LOSSESS = False
_C.SOLVER.IMPORTANCE_LOSS = False

_C.SOLVER.AUGLOSS = False
_C.SOLVER.AUGLOSS_START = -1
_C.SOLVER.AUGLOSS_INTERVAL = -1
_C.SOLVER.AUGLOSS_ENDITER = -1

_C.SOLVER.CROSS_LOSS = False


_C.SOLVER.LAYER_LR_DECAY = 1.0

_C.SOLVER.FORCE_SOFTMAX_FP16 = False
_C.SOLVER.FORCE_LN_FP16 = False
_C.SOLVER.FORCE_NORM_FP16 = False
_C.SOLVER.FORCE_TEMP_FP16 = False

_C.SOLVER.FORCE_WG_RECAST = False

_C.SOLVER.FORCE_EXPERT_ADDING_FP16 = False
_C.SOLVER.FORCE_EMBED_FP16 = False



# ----------------------------------------------------------------------------
# lr scheduler
# ----------------------------------------------------------------------------
_C.LR_SCHEDULER = CN()

_C.LR_SCHEDULER.NAME = 'StepLR'

_C.LR_SCHEDULER.STEP_SIZE = 3

_C.LR_SCHEDULER.GAMMA = 0.1

_C.LR_SCHEDULER.MODEL_SIZE = -1 # for Noam only

_C.LR_SCHEDULER.FACTOR = 1.0 # for Noam only

_C.LR_SCHEDULER.WARMUP = 1000 # epoch, for WarmupXXX; iteration, for Noam

_C.LR_SCHEDULER.MIN_LR = 0.000001

_C.LR_SCHEDULER.STEPS = (3,) # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_FACTOR = 0.0 # for WarmupMultiStep only

_C.LR_SCHEDULER.WARMUP_METHOD = "linear" # for WarmupMultiStep only
_C.LR_SCHEDULER.WARMUPTYPE = "linear"  # for WarmupMultiStep only

_C.LR_SCHEDULER.MILESTONES = []

# ---------------------------------------------------------------------------- #
# Losses
# ---------------------------------------------------------------------------- #
_C.LOSSES = CN()

_C.LOSSES.NAMES = ['']

_C.LOSSES.LOSS_WEIGHT = 1.0

_C.LOSSES.REDUCTION = 'mean'

_C.LOSSES.LABELSMOOTHING = 0.1

_C.LOSSES.MARGIN = 0.2

_C.LOSSES.LOSS_FP32 = False

_C.LOSSES.MAX_VIOLATION = True

# ---------------------------------------------------------------------------- #
# SCORER options
# ---------------------------------------------------------------------------- #
_C.SCORER = CN()

_C.SCORER.NAME = ''

_C.SCORER.TYPES = ['']

_C.SCORER.WEIGHTS = [1.0]

_C.SCORER.GT_PATH = 'coco_train_gts.pkl'

_C.SCORER.CIDER_CACHED = 'coco_train_cider.pkl'

_C.SCORER.EOS_ID = 0

# ---------------------------------------------------------------------------- #
# Decode strategy
# ---------------------------------------------------------------------------- #
_C.DECODE_STRATEGY = CN()

_C.DECODE_STRATEGY.NAME = 'none'

_C.DECODE_STRATEGY.BEAM_SIZE = 1

_C.DECODE_STRATEGY.LEN_PENALTY = 0.0

# ---------------------------------------------------------------------------- #
# INFERENCE options
# ---------------------------------------------------------------------------- #
_C.INFERENCE = CN()

_C.INFERENCE.NAME = ''

_C.INFERENCE.VOCAB = 'CLIP'

_C.INFERENCE.ID_KEY = 'image_id'

_C.INFERENCE.VALUE = 'caption'

_C.INFERENCE.VAL_ANNFILE = 'captions_val5k.json'

_C.INFERENCE.TEST_ANNFILE = 'captions_test5k.json'

_C.INFERENCE.GENERATION_MODE = True

_C.INFERENCE.VAL_EVAL_START = -1

_C.INFERENCE.TEST_EVAL_START = -1

_C.INFERENCE.ITER_BASED = True 

_C.INFERENCE.EVAL_BS = 100

# xiaoshi: added for video cls
_C.INFERENCE.NUM_VIEWS = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./output"

_C.SEED = -1

_C.CUDNN_BENCHMARK = False

_C.find_unused_parameters = True

_C.MOE = CN()

_C.MOE.MOE = False

_C.MOE.EP_WORLD_SIZE = 1

_C.MOE.NUM_EXPERTS =  1

_C.MOE.TOP_K = 1

_C.MOE.CAPACITY_FACTOR = 1.0

_C.MOE.EVAL_MIN_CAPACITY = 1.0

_C.MOE.MIN_CAPACITY = 4

_C.MOE.NOISY_GATE_POLICY = 'RSample'

_C.MOE.USE_RTS = True

_C.MOE.USE_TUTEL = False

_C.MOE.MOE_PARAM_GROUP = True

_C.MOE.MOE_EXPERT_TYPE = 'FFN'

_C.MOE.MOE_EXPERT_LOCATION = 'odd'

_C.MOE.SA_LINEAR_OUT_MOE = False

_C.MOE.KV_SHARED = False

_C.MOE.TASK_MOE = False

_C.MOE.CUSTOM_MOE = False

_C.MOE.MOE_TYPE = 'attribute'
_C.MOE.ATTRIBUTE_LENGTH = 8



_C.MOE.GATE_SOURCE = 'spe'
_C.MOE.LAUX_CONFIG = ''
_C.MOE.LAUX_ONEHOT = ''  # batchonehot sampleonehot
_C.MOE.LAUX_TYPE = 'std'  # batchonehot sampleonehot
_C.MOE.WORD_LAUX = 'even'  # onehot
_C.MOE.ATTENTION_OUT = 'mean'
_C.MOE.WORD_EXPERT_REGULARIZER = False

_C.MOE.MOE_LAYER_START_IDX = -1
_C.MOE.MOE_LAYER_END_IDX = 999
_C.MOE.BATCH_PRIO = False
_C.MOE.GATE_TYPE = 'deepspeed'
_C.MOE.LN_MOE = False
_C.MOE.FFN_SHARE_GATE_DECISION = False
_C.MOE.FFN_SA_SHARE_GATE = False
_C.MOE.FFN_MOE_SEPARATE = False
_C.MOE.MERGE_EXPERTS = False
_C.MOE.TAG_Transform = False
_C.MOE.TAG_Transform_ACT = False
_C.MOE.TAG_Transform_ALONE = False
_C.MOE.NOISE_STD = 1.0

_C.SOLVER.FLOPS_PROFILER = False

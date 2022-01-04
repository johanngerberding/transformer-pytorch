from yacs.config import CfgNode as CN 

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.TRAIN = CN()
_C.TRAIN.N_EPOCHS = 20
_C.TRAIN.BATCH_SIZE = 32

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 16

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "Adam"
_C.OPTIMIZER.INITIAL_LR = 0.0
_C.OPTIMIZER.CLIP = 1
_C.OPTIMIZER.BETA_1 = 0.9
_C.OPTIMIZER.BETA_2 = 0.98
_C.OPTIMIZER.EPS = 0.0000000001


_C.MODEL = CN()
_C.MODEL.ENC_HIDDEN_DIM = 512
_C.MODEL.DEC_HIDDEN_DIM = 512
_C.MODEL.ENC_LAYERS = 6
_C.MODEL.DEC_LAYERS = 6
_C.MODEL.ENC_HEADS = 8
_C.MODEL.DEC_HEADS = 8
_C.MODEL.ENC_PF_DIM = 512
_C.MODEL.DEC_PF_DIM = 512
_C.MODEL.ENC_DROPOUT = 0.1
_C.MODEL.DEC_DROPOUT = 0.1

_C.DATASET = CN()
_C.DATASET.NAME = "wmt14"
_C.DATASET.SRC_LANGUAGE = "de"
_C.DATASET.TGT_LANGUAGE = "en"
_C.DATASET.UNK_IDX = 0
_C.DATASET.PAD_IDX = 1
_C.DATASET.BOS_IDX = 2
_C.DATASET.EOS_IDX = 3
_C.DATASET.SPECIAL_SYMBOLS = ["unk", "pad", "bos", "eos"]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()
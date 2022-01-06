from yacs.config import CfgNode as CN 

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.TRAIN = CN()
_C.TRAIN.N_EPOCHS = 5
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.SMOOTHING = 0.1

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 32

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = "Adam"
_C.OPTIMIZER.INITIAL_LR = 0.0
_C.OPTIMIZER.BETA_1 = 0.9
_C.OPTIMIZER.BETA_2 = 0.98
_C.OPTIMIZER.EPS = 0.0000000001
_C.OPTIMIZER.WARMUP = 2000
_C.OPTIMIZER.FACTOR = 1


_C.MODEL = CN()
_C.MODEL.D_MODEL = 512
_C.MODEL.NUM_LAYERS = 6
_C.MODEL.ATTN_HEADS = 8
_C.MODEL.FFN_DIM = 2048
_C.MODEL.DROPOUT = 0.1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()
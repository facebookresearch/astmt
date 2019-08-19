from .bsds import BSDS500
from .coco import COCOSegmentation
from .fsv import FSVGTA
from .nyud import NYUD_MT, NYUDRaw
from .pascal_context import PASCALContext
from .pascal_voc import VOC12
from .sbd import SBD
from .msra10k import MSRA
from .pascal_sal import PASCALS

__all__ = ['BSDS500', 'COCOSegmentation', 'FSVGTA', 'NYUD_MT',
           'NYUDRaw', 'PASCALContext', 'VOC12', 'SBD', 'MSRA', 'PASCALS']
__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from . import SupervisedEvaluation, LargeVolumeEvaluation
from helper import argget


class ClassificationEvaluation(SupervisedEvaluation):
    def __init__(self, model, collectioninst, **kw):
        self.nclasses = argget(kw, 'nclasses', 2, keep=True)
        super(ClassificationEvaluation, self).__init__(model, collectioninst,
                                                       **kw)


class LargeVolumeClassificationEvaluation(LargeVolumeEvaluation, ClassificationEvaluation):
    pass
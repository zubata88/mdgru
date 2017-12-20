__author__ = "Simon Andermatt"
__copyright__ = "Copyright (C) 2017 Simon Andermatt"

from . import SupervisedEvaluation, LargeVolumeEvaluation
from helper import argget
import logging
import numpy as np
import os


class LocationClassificationEvaluation(SupervisedEvaluation):
    def __init__(self, model, collectioninst, **kw):
        self.nclasses = argget(kw, 'ncoords', [64, 64, 64], keep=True)
        super(LocationClassificationEvaluation, self).__init__(model, collectioninst,
                                                               **kw)


class LargeVolumeLocationCoordinateClassificationEvaluation(LocationClassificationEvaluation):
    def __init__(self, model, collectioninst, **kw):
        self.resultsfile = argget(kw, 'resultsfile', 'temp.csv')
        self.evaluate_uncertainty_times = argget(kw, "evaluate_uncertainty_times", 1)
        self.evaluate_uncertainty_dropout = argget(kw, "evaluate_uncertainty_dropout",
                                                   1.0)  # these standard values ensure that we dont evaluate uncertainty if nothing was provided.
        super(LargeVolumeLocationCoordinateClassificationEvaluation, self).__init__(model, collectioninst,
                                                                                    **kw)

    def test_all_available(self, batch_size=None, dc=None):
        if dc is None:
            dc = self.tedc
        if batch_size > 1:
            logging.getLogger('eval').error('not supported yet to have more than batchsize 1')
        additionalresultsfile = open(self.resultsfile + '-additional.csv', 'w')
        locs = "loc1{}loc2{}loc3{}".format(*["".join([',' for _ in range(n)]) for n in dc.ncoords])
        with open(self.resultsfile, 'w') as f:
            f.write("tp,loc1,loc2,loc3," + locs + "\n")
            f.write(",,,,{},{},{}\n".format(*[",".join(str(i) for i in range(n)) for n in dc.ncoords]))
            additionalresultsfile.write(
                'in the following, the extended results are displayed. first tp, then x,y,z, and then the full vectorial information of each test run per tp.\n')
            additionalresultsfile.write('tp,loc1,loc2,loc3')
            for _ in range(self.evaluate_uncertainty_times):
                additionalresultsfile.write(locs)
            additionalresultsfile.write("\n")
            for i in range(len(dc.tps)):
                batch, _, feature_padding = dc.sample_at(tp=i)
                tp = dc.tps[i]
                preds = []
                for _ in range(self.evaluate_uncertainty_times):
                    preds.append(self.sess.run(self.model.prediction,
                                               {self.data: batch, self.dropout: self.evaluate_uncertainty_dropout,
                                                self.training: False}))
                pred = np.mean(np.asarray(preds), 0)
                uncert = np.std(np.asarray(preds), 0)

                locs = np.argmax(pred[0]), np.argmax(pred[1]), np.argmax(pred[2])
                logging.getLogger('eval').info(','.join(str(l) for l in locs))

                scaling = [1.0 * w / d for w, d in zip(dc.w, dc.ncoords)]
                corrected_loc_diff = [l * s for l, s in zip(locs, scaling)]  # scale appropriately
                logging.getLogger('eval').info(','.join(str(l) for l in corrected_loc_diff))

                corrected_loc_diff = [l - (w // 2) for l, w in
                                      zip(corrected_loc_diff, dc.w)]  # move coordinates to center
                logging.getLogger('eval').info(','.join(str(l) for l in corrected_loc_diff))

                corrected_loc_diff = [l - (s // 2) for l, s in zip(corrected_loc_diff,
                                                                   feature_padding)]  # correct for too small full volume dimensions
                logging.getLogger('eval').info(','.join(str(l) for l in corrected_loc_diff))

                givencoarseloc = dc.labels[os.path.basename(tp)]
                logging.getLogger('eval').info(','.join(str(l) for l in givencoarseloc))

                x, y, z = [g + c for g, c in zip(givencoarseloc, corrected_loc_diff)]
                logging.getLogger('eval').info('tp{}x{}y{}z{}'.format(os.path.basename(tp), x, y, z))
                logging.getLogger('eval').info(
                    'std{},{},{}'.format(",".join(str(a) for a in pred[0]), ",".join(str(b) for b in pred[1]),
                                         ",".join(str(c) for c in pred[2])))

                f.write("{},{},{},{},{},{},{},{},{},{}\n".format(os.path.basename(tp), x, y, z,
                                                                 ",".join(str(a) for a in pred[0]),
                                                                 ",".join(str(b) for b in pred[1]),
                                                                 ",".join(str(c) for c in pred[2]),
                                                                 ",".join(str(a) for a in uncert[0]),
                                                                 ",".join(str(b) for b in uncert[1]),
                                                                 ",".join(str(c) for c in uncert[2])))
                additionalresultsfile.write("{},{},{},{}".format(os.path.basename(tp), x, y, z))
                for p in preds:
                    additionalresultsfile.write(
                        ",{},{},{}".format(",".join(str(a) for a in p[0]), ",".join(str(b) for b in p[1]),
                                           ",".join(str(c) for c in p[2]), ))
                additionalresultsfile.write("\n")
                logging.getLogger('eval').info(
                    '{},{},{},{},{},{},{},{},{},{}'.format(os.path.basename(tp), locs[0], locs[1], locs[2], x, y, z,
                                                           ",".join(str(a) for a in pred[0]),
                                                           ",".join(str(b) for b in pred[1]),
                                                           ",".join(str(c) for c in pred[2])))


class LargeVolumeLocationClassificationEvaluation(LargeVolumeEvaluation, LocationClassificationEvaluation):
    pass

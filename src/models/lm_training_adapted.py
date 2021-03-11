#!/usr/bin/env python
# -*- coding: utf-8 -*-

class MainTrainingClass(object):

    """main class to train models using the feature selection algorithm"""

    def __init__(self, model, params):
        """Define and initialize fields"""
        self.model = model
        self.params = params

    def classifier_fs(self, loss_tensor):
        """Implements the feature selection method

        Args:
            loss_tensor (tensor): Loss computed from the tf.keras model
        """
        pass
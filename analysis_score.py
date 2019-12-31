#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import coref_model_sentence_span as cm
import util

if __name__ == "__main__":
  os.environ["GPU"] = "0"
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  with tf.Session() as session:
    model.restore(session)
    model.analysis_top_score(session, official_stdout=True)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow as tf
import coref_model_sentence_span as cm
import util

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize


def create_example(text):
  raw_sentences = sent_tokenize(text)
  print('raw_sentence:', raw_sentences)
  sentences = [word_tokenize(s) for s in raw_sentences]
  print('sentences:', sentences)
  speakers = [["" for _ in sentence] for sentence in sentences]
  print('speakers:', speakers)

  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }


def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:

    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))


def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)

  print('tensorized_example', tensorized_example)
  # print('model.input_tensors', tensorized_example)

  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  # print('feed_dict', feed_dict)
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)
  # print("mention_starts", mention_starts, mention_starts.shape)
  # print("mention_ends", mention_ends, mention_starts.shape)
  # print("antecedents", antecedents, antecedents.shape)
  # print("antecedent_scores", antecedent_scores, antecedent_scores.shape)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)
  # print("predicted_antecedents", predicted_antecedents)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  # print('predicted_example:', example)
  return example


if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  with tf.Session() as session:
    model.restore(session)
    while True:
      text = input("Document text: ")
      if len(text) > 0:
        print_predictions(make_predictions(text, model))

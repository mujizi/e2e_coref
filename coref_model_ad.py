from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -*-encoding:utf-8-*-

import os
import operator
import random
import math
import json
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import util
import coref_ops
import conll
import metrics
import tools
import re

NUM = re.compile(r'\d{4}')


class CorefModel(object):
    def __init__(self, config):
        self.config = config
        self.context_embeddings = util.EmbeddingDictionary(config["context_embeddings"])
        self.head_embeddings = util.EmbeddingDictionary(config["head_embeddings"], maybe_cache=self.context_embeddings)
        self.char_embedding_size = config["char_embedding_size"]
        self.char_dict = util.load_char_dict(config["char_vocab_path"])
        self.max_span_width = config["max_span_width"]
        self.genres = {g: i for i, g in enumerate(config["genres"])}
        if config["lm_path"]:
            self.lm_file = h5py.File(self.config["lm_path"], "r")
        else:
            self.lm_file = None
        self.lm_layers = self.config["lm_layers"]
        self.lm_size = self.config["lm_size"]
        self.eval_data = None  # Load eval data lazily.

        input_props = []
        input_props.append((tf.string, [None, None]))  # Tokens.
        input_props.append((tf.float32, [None, None, self.context_embeddings.size]))  # Context embeddings.
        input_props.append((tf.float32, [None, None, self.head_embeddings.size]))  # Head embeddings.
        input_props.append((tf.float32, [None, None, self.lm_size, self.lm_layers]))  # LM embeddings.
        input_props.append((tf.int32, [None, None, None]))  # Character indices.
        input_props.append((tf.int32, [None]))  # Text lengths.
        input_props.append((tf.int32, [None]))  # Speaker IDs.
        input_props.append((tf.int32, []))  # Genre.
        input_props.append((tf.bool, []))  # Is training.
        input_props.append((tf.int32, [None]))  # Gold starts.
        input_props.append((tf.int32, [None]))  # Gold ends.
        input_props.append((tf.int32, [None]))  # Cluster ids.
        input_props.append((tf.int32, [None]))  # sentence_start.
        input_props.append((tf.int32, [None]))  # sentence_end.

        self.queue_input_tensors = [tf.placeholder(dtype, shape) for dtype, shape in input_props]
        dtypes, shapes = zip(*input_props)
        queue = tf.PaddingFIFOQueue(capacity=10, dtypes=dtypes, shapes=shapes)
        self.enqueue_op = queue.enqueue(self.queue_input_tensors)
        self.input_tensors = queue.dequeue()

        self.predictions, self.loss, self.top_list = self.get_predictions_and_loss(*self.input_tensors)

        """adversarial train, used FGM"""
        self.top_embed, top_ids, top_scores, top_speaker, genre_emb_r, k_r = self.top_list
        self.copy_top_embed = tf.identity(self.top_embed)

        with tf.name_scope('ad') as scope:
            self.ad_loss = self.adversarial_loss(self.copy_top_embed, top_ids, top_scores, top_speaker, genre_emb_r, k_r)
            gradients_r = tf.gradients(self.ad_loss, self.copy_top_embed, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            gradients_r = tf.stop_gradient(gradients_r)
            norm = tf.norm(gradients_r)
            r = gradients_r / norm
            self.copy_top_embed = self.copy_top_embed + r

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.reset_global_step = tf.assign(self.global_step, 0)
        learning_rate = tf.train.exponential_decay(self.config["learning_rate"], self.global_step,
                                                   self.config["decay_frequency"], self.config["decay_rate"],
                                                   staircase=True)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss * 0.6 + self.ad_loss * 0.4, trainable_params)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config["max_gradient_norm"])
        optimizers = {
            "adam": tf.train.AdamOptimizer,
            "sgd": tf.train.GradientDescentOptimizer
        }
        optimizer = optimizers[self.config["optimizer"]](learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, trainable_params), global_step=self.global_step)

    def start_enqueue_thread(self, session):
        with open(self.config["train_path"]) as f:
            train_examples = [json.loads(jsonline) for jsonline in f.readlines()]

        def _enqueue_loop():
            while True:
                random.shuffle(train_examples)
                for example in train_examples:
                    tensorized_example = self.tensorize_example(example, is_training=True)
                    feed_dict = dict(zip(self.queue_input_tensors, tensorized_example))
                    session.run(self.enqueue_op, feed_dict=feed_dict)

        enqueue_thread = threading.Thread(target=_enqueue_loop)
        enqueue_thread.daemon = True
        enqueue_thread.start()

    def restore(self, session):
        # Don't try to restore unused variables from the TF-Hub ELMo module.
        vars_to_restore = [v for v in tf.global_variables() if "module/" not in v.name]
        saver = tf.train.Saver(vars_to_restore)

        # checkpoint_path = os.path.join(self.config["log_dir"], "model.max.ckpt")
        checkpoint_path = os.path.join(self.config["log_dir"], "model-4500")

        print("Restoring from {}".format(checkpoint_path))
        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint_path)

    def load_lm_embeddings(self, doc_key):
        if self.lm_file is None:
            return np.zeros([0, 0, self.lm_size, self.lm_layers])
        file_key = doc_key.replace("/", ":")
        group = self.lm_file[file_key]
        num_sentences = len(list(group.keys()))
        sentences = [group[str(i)][...] for i in range(num_sentences)]
        lm_emb = np.zeros([num_sentences, max(s.shape[0] for s in sentences), self.lm_size, self.lm_layers])
        for i, s in enumerate(sentences):
            lm_emb[i, :s.shape[0], :, :] = s
        return lm_emb

    def sentence_start_end_index(self, sentences):
        """
        :param sentences: sentences list example: [['i', 'like'], ['cat']]
        :return: sentences start list and end list just like start:[0, 2], end:[1, 2]
        """
        start_l, end_l = [], []
        offset = -1
        for sentence in sentences:
            start_ = offset + 1
            end_ = len(sentence) + offset
            try:
                if sentence[0] == '[' and NUM.match(sentence[1]) and sentence[2] == ']':
                    start_ = start_ + 3
            finally:
                offset = offset + len(sentence)
                if abs(end_ - start_ + 1) > 30:
                    start_l.append(start_)
                    end_l.append(end_)
        assert len(start_l) == len(end_l)
        return np.array(start_l), np.array(end_l)

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
            starts, ends = zip(*mentions)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def tensorize_span_labels(self, tuples, label_dict):
        if len(tuples) > 0:
            starts, ends, labels = zip(*tuples)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[c] for c in labels])

    def tensorize_example(self, example, is_training):
        clusters = example["clusters"]
        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        # print('gold_mentions', gold_mentions)

        gold_mention_map = {m: i for i, m in enumerate(gold_mentions)}
        # print('gold_mention_map', gold_mention_map)

        cluster_ids = np.zeros(len(gold_mentions))
        # print(cluster_ids)

        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
        # print('cluster_ids', cluster_ids)

        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = util.flatten(example["speakers"])

        assert num_words == len(speakers)

        max_sentence_length = max(len(s) for s in sentences)
        max_word_length = max(max(max(len(w) for w in s) for s in sentences), max(self.config["filter_widths"]))
        text_len = np.array([len(s) for s in sentences])
        tokens = [[""] * max_sentence_length for _ in sentences]

        context_word_emb = np.zeros([len(sentences), max_sentence_length, self.context_embeddings.size])
        # print('context_word_emb', context_word_emb, context_word_emb.shape)

        head_word_emb = np.zeros([len(sentences), max_sentence_length, self.head_embeddings.size])
        # print('head_word_emb', head_word_emb, head_word_emb.shape)

        char_index = np.zeros([len(sentences), max_sentence_length, max_word_length])
        # print('char_index', char_index, char_index.shape)

        for i, sentence in enumerate(sentences):
            for j, word in enumerate(sentence):
                tokens[i][j] = word
                context_word_emb[i, j] = self.context_embeddings[word]

                head_word_emb[i, j] = self.head_embeddings[word]
                char_index[i, j, :len(word)] = [self.char_dict[c] for c in word]
        tokens = np.array(tokens)

        # print('context_word_emb', context_word_emb)

        speaker_dict = {s: i for i, s in enumerate(set(speakers))}
        speaker_ids = np.array([speaker_dict[s] for s in speakers])

        doc_key = example["doc_key"]
        genre = self.genres[doc_key[:2]]

        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        sentence_index_start, sentence_index_end = self.sentence_start_end_index(sentences)

        lm_emb = self.load_lm_embeddings(doc_key)

        # example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids)
        example_tensors = (tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training,
        gold_starts, gold_ends, cluster_ids, sentence_index_start, sentence_index_end)

        if is_training and len(sentences) > self.config["max_training_sentences"]:
            return self.truncate_example(*example_tensors)
        else:
            return example_tensors

    def truncate_example(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids,
                         genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_index_start,
                         sentence_index_end):
        max_training_sentences = self.config["max_training_sentences"]
        num_sentences = context_word_emb.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences)
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        tokens = tokens[sentence_offset:sentence_offset + max_training_sentences, :]
        context_word_emb = context_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        head_word_emb = head_word_emb[sentence_offset:sentence_offset + max_training_sentences, :, :]
        lm_emb = lm_emb[sentence_offset:sentence_offset + max_training_sentences, :, :, :]
        char_index = char_index[sentence_offset:sentence_offset + max_training_sentences, :, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        speaker_ids = speaker_ids[word_offset: word_offset + num_words]

        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)

        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset

        """i want to get sentence start and end index
        """
        sentence_spans = np.logical_and(sentence_index_end >= word_offset,
                                        sentence_index_start < word_offset + num_words)
        sentence_index_start = sentence_index_start[sentence_spans] - word_offset
        sentence_index_end = sentence_index_end[sentence_spans] - word_offset

        cluster_ids = cluster_ids[gold_spans]

        return tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_index_start, sentence_index_end

    def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
        same_start = tf.equal(tf.expand_dims(labeled_starts, 1),
                              tf.expand_dims(candidate_starts, 0))  # [num_labeled, num_candidates]
        same_end = tf.equal(tf.expand_dims(labeled_ends, 1),
                            tf.expand_dims(candidate_ends, 0))  # [num_labeled, num_candidates]
        same_span = tf.logical_and(same_start, same_end)  # [num_labeled, num_candidates]
        candidate_labels = tf.matmul(tf.expand_dims(labels, 0), tf.to_int32(same_span))  # [1, num_candidates]
        candidate_labels = tf.squeeze(candidate_labels, 0)  # [num_candidates]
        return candidate_labels

    def get_dropout(self, dropout_rate, is_training):
        return 1 - (tf.to_float(is_training) * dropout_rate)

    def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_span_range = tf.range(k)  # [k]
        antecedent_offsets = tf.expand_dims(top_span_range, 1) - tf.expand_dims(top_span_range, 0)  # [k, k]
        antecedents_mask = antecedent_offsets >= 1  # [k, k]
        fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.expand_dims(top_span_mention_scores,
                                                                                             0)  # [k, k]
        fast_antecedent_scores += tf.log(tf.to_float(antecedents_mask))  # [k, k]
        fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb)  # [k, k]

        _, top_antecedents = tf.nn.top_k(fast_antecedent_scores, c, sorted=False)  # [k, c]
        top_antecedents_mask = util.batch_gather(antecedents_mask, top_antecedents)  # [k, c]
        top_fast_antecedent_scores = util.batch_gather(fast_antecedent_scores, top_antecedents)  # [k, c]
        top_antecedent_offsets = util.batch_gather(antecedent_offsets, top_antecedents)  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def distance_pruning(self, top_span_emb, top_span_mention_scores, c):
        k = util.shape(top_span_emb, 0)
        top_antecedent_offsets = tf.tile(tf.expand_dims(tf.range(c) + 1, 0), [k, 1])  # [k, c]
        raw_top_antecedents = tf.expand_dims(tf.range(k), 1) - top_antecedent_offsets  # [k, c]
        top_antecedents_mask = raw_top_antecedents >= 0  # [k, c]
        top_antecedents = tf.maximum(raw_top_antecedents, 0)  # [k, c]

        top_fast_antecedent_scores = tf.expand_dims(top_span_mention_scores, 1) + tf.gather(top_span_mention_scores,
                                                                                            top_antecedents)  # [k, c]
        top_fast_antecedent_scores += tf.log(tf.to_float(top_antecedents_mask))  # [k, c]
        return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets

    def get_predictions_and_loss(self, tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len,
                                 speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids,
                                 sentence_index_start, sentence_index_end):
        self.dropout = self.get_dropout(self.config["dropout_rate"], is_training)
        self.lexical_dropout = self.get_dropout(self.config["lexical_dropout_rate"], is_training)
        self.lstm_dropout = self.get_dropout(self.config["lstm_dropout_rate"], is_training)

        num_sentences = tf.shape(context_word_emb)[0]
        max_sentence_length = tf.shape(context_word_emb)[1]

        context_emb_list = [context_word_emb]
        head_emb_list = [head_word_emb]

        if self.config["char_embedding_size"] > 0:
            char_emb = tf.gather(
                tf.get_variable("char_embeddings", [len(self.char_dict), self.config["char_embedding_size"]]),
                char_index)  # [num_sentences, max_sentence_length, max_word_length, emb]
            flattened_char_emb = tf.reshape(char_emb, [num_sentences * max_sentence_length, util.shape(char_emb, 2),
                                                       util.shape(char_emb,
                                                                  3)])  # [num_sentences * max_sentence_length, max_word_length, emb]
            flattened_aggregated_char_emb = util.cnn(flattened_char_emb, self.config["filter_widths"], self.config[
                "filter_size"])  # [num_sentences * max_sentence_length, emb]
            aggregated_char_emb = tf.reshape(flattened_aggregated_char_emb, [num_sentences, max_sentence_length,
                                                                             util.shape(flattened_aggregated_char_emb,
                                                                                        1)])  # [num_sentences, max_sentence_length, emb]
            context_emb_list.append(aggregated_char_emb)
            head_emb_list.append(aggregated_char_emb)

        if not self.lm_file:
            elmo_module = hub.Module("https://tfhub.dev/google/elmo/2")
            lm_embeddings = elmo_module(
                inputs={"tokens": tokens, "sequence_len": text_len},
                signature="tokens", as_dict=True)
            word_emb = lm_embeddings["word_emb"]  # [num_sentences, max_sentence_length, 512]
            lm_emb = tf.stack([tf.concat([word_emb, word_emb], -1),
                               lm_embeddings["lstm_outputs1"],
                               lm_embeddings["lstm_outputs2"]], -1)  # [num_sentences, max_sentence_length, 1024, 3]
        lm_emb_size = util.shape(lm_emb, 2)
        lm_num_layers = util.shape(lm_emb, 3)
        with tf.variable_scope("lm_aggregation"):
            self.lm_weights = tf.nn.softmax(
                tf.get_variable("lm_scores", [lm_num_layers], initializer=tf.constant_initializer(0.0)))
            self.lm_scaling = tf.get_variable("lm_scaling", [], initializer=tf.constant_initializer(1.0))
        flattened_lm_emb = tf.reshape(lm_emb, [num_sentences * max_sentence_length * lm_emb_size, lm_num_layers])
        flattened_aggregated_lm_emb = tf.matmul(flattened_lm_emb, tf.expand_dims(self.lm_weights,
                                                                                 1))  # [num_sentences * max_sentence_length * emb, 1]
        aggregated_lm_emb = tf.reshape(flattened_aggregated_lm_emb, [num_sentences, max_sentence_length, lm_emb_size])
        aggregated_lm_emb *= self.lm_scaling
        context_emb_list.append(aggregated_lm_emb)

        context_emb = tf.concat(context_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.concat(head_emb_list, 2)  # [num_sentences, max_sentence_length, emb]
        context_emb = tf.nn.dropout(context_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]
        head_emb = tf.nn.dropout(head_emb, self.lexical_dropout)  # [num_sentences, max_sentence_length, emb]

        """ get context embedding with glove 300 and 50, the 300 dimension embedding use to concatenate char embedding and other
        feature embedding. And the 50 dimension embedding with glove_300d_2w use to compute attention. They are encoding with 
        context.
    
        1. context whole embedding equal language model elmo get token embedding(aggregated_lm_emb) concatenate char embedding
        concatenate glove(300 dimension word2vec embedding).
        shape: [num_sentences, max_sentence_length, emb]
    
        2. head_emb equal glove(50 dimension word2vec embedding concatenate char embedding)
        shape: [num_sentences, max_sentence_length, emb]
        """

        text_len_mask = tf.sequence_mask(text_len, maxlen=max_sentence_length)  # [num_sentence, max_sentence_length]
        context_outputs = self.lstm_contextualize(context_emb, text_len, text_len_mask)  # [num_words, emb]

        """Used context lstm encoder
        input: context whole embedding  
        output: context_output 
        shape: [num_sentences * max_sentence_length, emb]
        """

        num_words = util.shape(context_outputs, 0)

        genre_emb = tf.gather(tf.get_variable("genre_embeddings", [len(self.genres), self.config["feature_size"]]),
                              genre)  # [emb]

        sentence_indices = tf.tile(tf.expand_dims(tf.range(num_sentences), 1),
                                   [1, max_sentence_length])  # [num_sentences, max_sentence_length]
        flattened_sentence_indices = self.flatten_emb_by_sentence(sentence_indices, text_len_mask)  # [num_words]
        flattened_head_emb = self.flatten_emb_by_sentence(head_emb, text_len_mask)  # [num_words]

        candidate_starts = tf.tile(tf.expand_dims(tf.range(num_words), 1),
                                   [1, self.max_span_width])  # [num_words, max_span_width]
        candidate_ends = candidate_starts + tf.expand_dims(tf.range(self.max_span_width),
                                                           0)  # [num_words, max_span_width]

        """ add sentence candidate starts and candidate ends in this place. and padding max_span_width to max_sentence_len.
            sentence_indices not used in this model input
        """
        # print('candidate_starts_1', candidate_starts)

        candidate_start_sentence_indices = tf.gather(flattened_sentence_indices,
                                                     candidate_starts)  # [num_words, max_span_width]
        candidate_end_sentence_indices = tf.gather(flattened_sentence_indices, tf.minimum(candidate_ends,
                                                                                          num_words - 1))  # [num_words, max_span_width]
        candidate_mask = tf.logical_and(candidate_ends < num_words, tf.equal(candidate_start_sentence_indices,
                                                                             candidate_end_sentence_indices))  # [num_words, max_span_width]
        flattened_candidate_mask = tf.reshape(candidate_mask, [-1])  # [num_words * max_span_width]
        candidate_starts = tf.boolean_mask(tf.reshape(candidate_starts, [-1]),
                                           flattened_candidate_mask)  # [num_candidates]
        candidate_ends = tf.boolean_mask(tf.reshape(candidate_ends, [-1]), flattened_candidate_mask)  # [num_candidates]
        candidate_sentence_indices = tf.boolean_mask(tf.reshape(candidate_start_sentence_indices, [-1]),
                                                     flattened_candidate_mask)  # [num_candidates]

        """this is my change in input queue, add sentence_index start and end in candidate. we want to add sentence level
        span candidate.
        """

        candidate_starts = tf.concat([candidate_starts, sentence_index_start], axis=0)
        candidate_ends = tf.concat([candidate_ends, sentence_index_end], axis=0)

        """think of use padding to change the span embedding dimention in this place.
    
        candidate_cluster_ids compare between candidate and gold mention,example second
        candidate is true, candidate_cluster_ids just like: [0, 1, 0]
    
    
        """

        candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends,
                                                          cluster_ids)  # [num_candidates]
        candidate_span_emb = self.get_span_emb(flattened_head_emb, context_outputs, candidate_starts,
                                               candidate_ends)  # [num_candidates, emb]
        candidate_mention_scores = self.get_mention_scores(candidate_span_emb)  # [k, 1]
        candidate_mention_scores = tf.squeeze(candidate_mention_scores, 1)  # [k]

        k = tf.to_int32(tf.floor(tf.to_float(tf.shape(context_outputs)[0]) * self.config["top_span_ratio"]))
        top_span_indices = coref_ops.extract_spans(tf.expand_dims(candidate_mention_scores, 0),
                                                   tf.expand_dims(candidate_starts, 0),
                                                   tf.expand_dims(candidate_ends, 0),
                                                   tf.expand_dims(k, 0),
                                                   util.shape(context_outputs, 0),
                                                   True)  # [1, k]
        top_span_indices.set_shape([1, None])
        top_span_indices = tf.squeeze(top_span_indices, 0)  # [k]

        top_span_starts = tf.gather(candidate_starts, top_span_indices)  # [k]
        top_span_ends = tf.gather(candidate_ends, top_span_indices)  # [k]
        top_span_emb = tf.gather(candidate_span_emb, top_span_indices)  # [k, emb]
        top_span_cluster_ids = tf.gather(candidate_cluster_ids, top_span_indices)  # [k]
        top_span_mention_scores = tf.gather(candidate_mention_scores, top_span_indices)  # [k]
        top_span_sentence_indices = tf.gather(candidate_sentence_indices, top_span_indices)  # [k]
        top_span_speaker_ids = tf.gather(speaker_ids, top_span_starts)  # [k]

        c = tf.minimum(self.config["max_top_antecedents"], k)

        """Stage 1 competed: k candidate mentions.
        """

        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
                top_span_emb, top_span_mention_scores, c)

        """Stage 2 competed: get each of k mensions c antecedents 
        shape: [k, c]   
        """

        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                     top_antecedents,
                                                                                                     top_antecedent_emb,
                                                                                                     top_antecedent_offsets,
                                                                                                     top_span_speaker_ids,
                                                                                                     genre_emb)  # [k, c]
                top_antecedent_weights = tf.nn.softmax(
                    tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                               1)  # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                  1)  # [k, emb]
                with tf.variable_scope("f"):
                    f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                   util.shape(top_span_emb, -1)))  # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        """Stage 3 used original paper section 3 function: the antecedent and top_span composed a pairs (coref entity, demonstraction
        pronoun) and computer the pair of score s(gi, gj), s(gi, gj) = top_fast_antecedent_scores + get_slow_antecdents, via softmax,
        get the weights of each k span's (c + 1) antecedents weight. P(yi), yi is i mention in top_span. This is a attention mechanism
        get a new embedding ai, ai are calculate by attention mechanism. And then concatenate ai and gi. matmul W and via sigmoid to 
        get a gatekeeper(fi). Finally, gi_final = fi * gi + (1 - fi) * ai. 
        shape: [k, emb]
        """

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]

        # print('top_antecedent_scores', top_antecedent_scores, top_antecedent_scores.shape)
        # print('top_antecedent_labels', top_antecedent_labels, top_antecedent_labels.shape)

        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]

        """result of k antecedents's softmax loss. 
        shape: [k]
        """

        loss = tf.reduce_sum(loss)  # []

        return [candidate_span_emb, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedents, top_antecedent_scores], loss, [top_span_emb, top_span_cluster_ids, top_span_mention_scores, top_span_speaker_ids,
                                                                genre_emb, k]

    def adversarial_loss(self, top_span_emb, top_span_cluster_ids, top_span_mention_scores, top_span_speaker_ids, genre_emb, k):
        c = tf.minimum(self.config["max_top_antecedents"], k)

        """Stage 1 competed: k candidate mentions.
        """

        if self.config["coarse_to_fine"]:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(
                top_span_emb, top_span_mention_scores, c)
        else:
            top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.distance_pruning(
                top_span_emb, top_span_mention_scores, c)

        """Stage 2 competed: get each of k mensions c antecedents 
        shape: [k, c]   
        """

        dummy_scores = tf.zeros([k, 1])  # [k, 1]
        for i in range(self.config["coref_depth"]):
            with tf.variable_scope("coref_layer", reuse=(i > 0)):
                top_antecedent_emb = tf.gather(top_span_emb, top_antecedents)  # [k, c, emb]
                top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb,
                                                                                                     top_antecedents,
                                                                                                     top_antecedent_emb,
                                                                                                     top_antecedent_offsets,
                                                                                                     top_span_speaker_ids,
                                                                                                     genre_emb)  # [k, c]
                top_antecedent_weights = tf.nn.softmax(
                    tf.concat([dummy_scores, top_antecedent_scores], 1))  # [k, c + 1]
                top_antecedent_emb = tf.concat([tf.expand_dims(top_span_emb, 1), top_antecedent_emb],
                                               1)  # [k, c + 1, emb]
                attended_span_emb = tf.reduce_sum(tf.expand_dims(top_antecedent_weights, 2) * top_antecedent_emb,
                                                  1)  # [k, emb]
                with tf.variable_scope("f"):
                    f = tf.sigmoid(util.projection(tf.concat([top_span_emb, attended_span_emb], 1),
                                                   util.shape(top_span_emb, -1)))  # [k, emb]
                    top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb  # [k, emb]

        """Stage 3 used original paper section 3 function: the antecedent and top_span composed a pairs (coref entity, demonstraction
        pronoun) and computer the pair of score s(gi, gj), s(gi, gj) = top_fast_antecedent_scores + get_slow_antecdents, via softmax,
        get the weights of each k span's (c + 1) antecedents weight. P(yi), yi is i mention in top_span. This is a attention mechanism
        get a new embedding ai, ai are calculate by attention mechanism. And then concatenate ai and gi. matmul W and via sigmoid to 
        get a gatekeeper(fi). Finally, gi_final = fi * gi + (1 - fi) * ai. 
        shape: [k, emb]
        """

        top_antecedent_scores = tf.concat([dummy_scores, top_antecedent_scores], 1)  # [k, c + 1]

        top_antecedent_cluster_ids = tf.gather(top_span_cluster_ids, top_antecedents)  # [k, c]
        top_antecedent_cluster_ids += tf.to_int32(tf.log(tf.to_float(top_antecedents_mask)))  # [k, c]
        same_cluster_indicator = tf.equal(top_antecedent_cluster_ids, tf.expand_dims(top_span_cluster_ids, 1))  # [k, c]
        non_dummy_indicator = tf.expand_dims(top_span_cluster_ids > 0, 1)  # [k, 1]
        pairwise_labels = tf.logical_and(same_cluster_indicator, non_dummy_indicator)  # [k, c]
        dummy_labels = tf.logical_not(tf.reduce_any(pairwise_labels, 1, keepdims=True))  # [k, 1]
        top_antecedent_labels = tf.concat([dummy_labels, pairwise_labels], 1)  # [k, c + 1]

        # print('top_antecedent_scores', top_antecedent_scores, top_antecedent_scores.shape)
        # print('top_antecedent_labels', top_antecedent_labels, top_antecedent_labels.shape)

        loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels)  # [k]

        """result of k antecedents's softmax loss. 
        shape: [k]
        """

        loss = tf.reduce_sum(loss)  # []
        return loss

    def dget_span_emb(self, head_emb, context_outputs, span_starts, span_ends):
        span_emb_list = []

        span_start_emb = tf.gather(context_outputs, span_starts)  # [k, emb]
        span_emb_list.append(span_start_emb)

        span_end_emb = tf.gather(context_outputs, span_ends)  # [k, emb]
        span_emb_list.append(span_end_emb)

        span_width = 1 + span_ends - span_starts  # [k]

        if self.config["use_features"]:
            span_width_index = tf.minimum(self.config["max_sentence_width"] - 1, span_width - 1)  # [k]
            span_width_emb = tf.gather(tf.get_variable("span_width_embeddings", [self.config["max_sentence_width"],
                                                                                 self.config["feature_size"]]),
                                       span_width_index)  # [k, emb]
            span_width_emb = tf.nn.dropout(span_width_emb, self.dropout)
            span_emb_list.append(span_width_emb)

            """
            In this place, i want to padding the max_span len to compute head attention span embedding.
      
            """
        if self.config["model_heads"]:
            span_indices = tf.expand_dims(tf.range(self.config["max_sentence_width"]), 0) + tf.expand_dims(span_starts,
                                                                                                           1)  # [k, max_span_width]
            span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
            span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]

            with tf.variable_scope("head_scores"):
                self.head_scores = util.projection(context_outputs, 1)  # [num_words, 1]

            span_head_scores = tf.gather(self.head_scores, span_indices)  # [k, max_span_width, 1]
            span_mask = tf.expand_dims(
                tf.sequence_mask(span_width, self.config["max_sentence_width"], dtype=tf.float32),
                2)  # [k, max_span_width, 1]
            span_head_scores += tf.log(span_mask)  # [k, max_span_width, 1]
            span_attention = tf.nn.softmax(span_head_scores, 1)  # [k, max_span_width, 1]
            span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1)  # [k, emb]
            span_emb_list.append(span_head_emb)

        # if self.config["model_heads"]:
        #   span_indices = tf.expand_dims(tf.range(self.config["max_span_width"]), 0) + tf.expand_dims(span_starts,
        #                                                                                              1)  # [k, max_span_width]
        #   span_indices = tf.minimum(util.shape(context_outputs, 0) - 1, span_indices)  # [k, max_span_width]
        #   span_text_emb = tf.gather(head_emb, span_indices)  # [k, max_span_width, emb]
        #
        #   with tf.variable_scope("head_scores"):
        #     self.head_scores = util.projection(context_outputs, 1) # [num_words, 1]
        #   span_head_scores = tf.gather(self.head_scores, span_indices) # [k, max_span_width, 1]
        #   span_mask = tf.expand_dims(tf.sequence_mask(span_width, self.config["max_span_width"], dtype=tf.float32), 2) # [k, max_span_width, 1]
        #   span_head_scores += tf.log(span_mask) # [k, max_span_width, 1]
        #   span_attention = tf.nn.softmax(span_head_scores, 1) # [k, max_span_width, 1]
        #   span_head_emb = tf.reduce_sum(span_attention * span_text_emb, 1) # [k, emb]
        #   span_emb_list.append(span_head_emb)

        span_emb = tf.concat(span_emb_list, 1)  # [k, emb]
        return span_emb  # [k, emb]

    def get_mention_scores(self, span_emb):
        with tf.variable_scope("mention_scores"):
            return util.ffnn(span_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1, self.dropout)  # [k, 1]

    def softmax_loss(self, antecedent_scores, antecedent_labels):

        # print('antecedent_scores', antecedent_scores, antecedent_scores.shape)
        # print('antecedent_label', antecedent_labels, antecedent_labels.shape)

        gold_scores = antecedent_scores + tf.log(tf.to_float(antecedent_labels))  # [k, max_ant + 1]
        marginalized_gold_scores = tf.reduce_logsumexp(gold_scores, [1])  # [k]
        log_norm = tf.reduce_logsumexp(antecedent_scores, [1])  # [k]

        # print('marginalized_gold_scores', marginalized_gold_scores)
        # print('log_norm', log_norm)

        return log_norm - marginalized_gold_scores  # [k]

    def bucket_distance(self, distances):
        """
        Places the given values (designed for distances) into 10 semi-logscale buckets:
        [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
        """
        logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
        use_identity = tf.to_int32(distances <= 4)
        combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
        return tf.clip_by_value(combined_idx, 0, 9)
        # return tf.clip_by_value(combined_idx, 0, 12)   # 256+

    def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets,
                                   top_span_speaker_ids, genre_emb):
        k = util.shape(top_span_emb, 0)
        c = util.shape(top_antecedents, 1)

        feature_emb_list = []

        if self.config["use_metadata"]:
            top_antecedent_speaker_ids = tf.gather(top_span_speaker_ids, top_antecedents)  # [k, c]
            same_speaker = tf.equal(tf.expand_dims(top_span_speaker_ids, 1), top_antecedent_speaker_ids)  # [k, c]
            speaker_pair_emb = tf.gather(tf.get_variable("same_speaker_emb", [2, self.config["feature_size"]]),
                                         tf.to_int32(same_speaker))  # [k, c, emb]
            feature_emb_list.append(speaker_pair_emb)

            tiled_genre_emb = tf.tile(tf.expand_dims(tf.expand_dims(genre_emb, 0), 0), [k, c, 1])  # [k, c, emb]
            feature_emb_list.append(tiled_genre_emb)

        if self.config["use_features"]:
            """ i think of that if want to increase the distance of cluster pair, we can change the [10] antecedent_distance_emb,
            and the bucket_distance function.
            """

            antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets)  # [k, c]
            antecedent_distance_emb = tf.gather(
                tf.get_variable("antecedent_distance_emb", [10, self.config["feature_size"]]),
                antecedent_distance_buckets)  # [k, c]
            # antecedent_distance_emb = tf.gather(tf.get_variable("antecedent_distance_emb", [13, self.config["feature_size"]]), antecedent_distance_buckets)  # [k, c]
            feature_emb_list.append(antecedent_distance_emb)

        feature_emb = tf.concat(feature_emb_list, 2)  # [k, c, emb]
        feature_emb = tf.nn.dropout(feature_emb, self.dropout)  # [k, c, emb]

        target_emb = tf.expand_dims(top_span_emb, 1)  # [k, 1, emb]
        similarity_emb = top_antecedent_emb * target_emb  # [k, c, emb]
        target_emb = tf.tile(target_emb, [1, c, 1])  # [k, c, emb]

        pair_emb = tf.concat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)  # [k, c, emb]

        with tf.variable_scope("slow_antecedent_scores"):
            slow_antecedent_scores = util.ffnn(pair_emb, self.config["ffnn_depth"], self.config["ffnn_size"], 1,
                                               self.dropout)  # [k, c, 1]
        slow_antecedent_scores = tf.squeeze(slow_antecedent_scores, 2)  # [k, c]
        return slow_antecedent_scores  # [k, c]

    def get_fast_antecedent_scores(self, top_span_emb):
        with tf.variable_scope("src_projection"):
            source_top_span_emb = tf.nn.dropout(util.projection(top_span_emb, util.shape(top_span_emb, -1)),
                                                self.dropout)  # [k, emb]
        target_top_span_emb = tf.nn.dropout(top_span_emb, self.dropout)  # [k, emb]
        return tf.matmul(source_top_span_emb, target_top_span_emb, transpose_b=True)  # [k, k]

    def flatten_emb_by_sentence(self, emb, text_len_mask):
        num_sentences = tf.shape(emb)[0]
        max_sentence_length = tf.shape(emb)[1]

        emb_rank = len(emb.get_shape())
        if emb_rank == 2:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length])
        elif emb_rank == 3:
            flattened_emb = tf.reshape(emb, [num_sentences * max_sentence_length, util.shape(emb, 2)])
        else:
            raise ValueError("Unsupported rank: {}".format(emb_rank))
        return tf.boolean_mask(flattened_emb, tf.reshape(text_len_mask, [num_sentences * max_sentence_length]))

    def lstm_contextualize(self, text_emb, text_len, text_len_mask):
        num_sentences = tf.shape(text_emb)[0]

        current_inputs = text_emb  # [num_sentences, max_sentence_length, emb]

        for layer in range(self.config["contextualization_layers"]):
            with tf.variable_scope("layer_{}".format(layer)):
                with tf.variable_scope("fw_cell"):
                    cell_fw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                with tf.variable_scope("bw_cell"):
                    cell_bw = util.CustomLSTMCell(self.config["contextualization_size"], num_sentences,
                                                  self.lstm_dropout)
                state_fw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_fw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_fw.initial_state.h, [num_sentences, 1]))
                state_bw = tf.contrib.rnn.LSTMStateTuple(tf.tile(cell_bw.initial_state.c, [num_sentences, 1]),
                                                         tf.tile(cell_bw.initial_state.h, [num_sentences, 1]))

                (fw_outputs, bw_outputs), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw,
                    cell_bw=cell_bw,
                    inputs=current_inputs,
                    sequence_length=text_len,
                    initial_state_fw=state_fw,
                    initial_state_bw=state_bw)

                text_outputs = tf.concat([fw_outputs, bw_outputs], 2)  # [num_sentences, max_sentence_length, emb]
                text_outputs = tf.nn.dropout(text_outputs, self.lstm_dropout)
                if layer > 0:
                    highway_gates = tf.sigmoid(util.projection(text_outputs, util.shape(text_outputs,
                                                                                        2)))  # [num_sentences, max_sentence_length, emb]
                    text_outputs = highway_gates * text_outputs + (1 - highway_gates) * current_inputs
                current_inputs = text_outputs

        return self.flatten_emb_by_sentence(text_outputs, text_len_mask)

    def get_predicted_antecedents(self, antecedents, antecedent_scores):
        predicted_antecedents = []
        for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if index < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedents[i, index])
        return predicted_antecedents

    def get_predicted_clusters(self, top_span_starts, top_span_ends, predicted_antecedents):
        mention_to_predicted = {}
        predicted_clusters = []
        for i, predicted_index in enumerate(predicted_antecedents):
            if predicted_index < 0:
                continue
            assert i > predicted_index
            predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
            if predicted_antecedent in mention_to_predicted:
                predicted_cluster = mention_to_predicted[predicted_antecedent]
            else:
                predicted_cluster = len(predicted_clusters)
                predicted_clusters.append([predicted_antecedent])
                mention_to_predicted[predicted_antecedent] = predicted_cluster

            mention = (int(top_span_starts[i]), int(top_span_ends[i]))
            predicted_clusters[predicted_cluster].append(mention)
            mention_to_predicted[mention] = predicted_cluster

        predicted_clusters = [tuple(pc) for pc in predicted_clusters]
        mention_to_predicted = {m: predicted_clusters[i] for m, i in mention_to_predicted.items()}

        return predicted_clusters, mention_to_predicted

    def evaluate_coref(self, top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
        gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
        mention_to_gold = {}
        for gc in gold_clusters:
            for mention in gc:
                mention_to_gold[mention] = gc

        predicted_clusters, mention_to_predicted = self.get_predicted_clusters(top_span_starts, top_span_ends,
                                                                               predicted_antecedents)
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters

    def load_eval_data(self):
        if self.eval_data is None:
            def load_line(line):
                example = json.loads(line)
                return self.tensorize_example(example, is_training=False), example

            with open(self.config["eval_path"]) as f:
                self.eval_data = [load_line(l) for l in f.readlines()]
            num_words = sum(tensorized_example[2].sum() for tensorized_example, _ in self.eval_data)

            print("Loaded {} eval examples.".format(len(self.eval_data)))

    def evaluate(self, session, official_stdout=False):
        self.load_eval_data()

        coref_predictions = {}
        coref_evaluator = metrics.CorefEvaluator()

        for example_num, (tensorized_example, example) in enumerate(self.eval_data):
            tokens, context_word_emb, head_word_emb, lm_emb, char_index, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, _, sentence_index_start, sentence_index_end = tensorized_example

            # print('tokens', tokens, tokens.shape)
            # print('context_word_emb', context_word_emb, context_word_emb.shape)
            # print('head_word_emb', head_word_emb, head_word_emb.shape)
            # print('lm', lm_emb, lm_emb.shape)

            feed_dict = {i: t for i, t in zip(self.input_tensors, tensorized_example)}
            candidate_span_emb, candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(
                self.predictions, feed_dict=feed_dict)

            # print('candidate_starts', candidate_starts)
            # print('end', candidate_ends)
            # print('top_span_starts', top_span_starts, top_span_ends.shape)
            # print('top_span_end', top_span_ends, top_span_ends.shape)
            # print('top_antecedent_scores', top_antecedent_scores)

            predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)

            # print('predicted_antecedents', predicted_antecedents)

            coref_predictions[example["doc_key"]] = self.evaluate_coref(top_span_starts, top_span_ends,
                                                                        predicted_antecedents, example["clusters"],
                                                                        coref_evaluator)
            if example_num % 10 == 0:
                print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))

        print('coref_predictions:', coref_predictions, len(coref_predictions))
        tools.write_json('/content/drive/My Drive/coreference/e2e/e2e-coref/coref_predictions.json',
                         coref_predictions)

        """this evaluation code is used to solve CoNLL style dataset evaluetions."""
        summary_dict = {}
        conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
        average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
        # summary_dict["Average F1 (conll)"] = average_f1
        # print("Average F1 (conll): {:.2f}%".format(average_f1))

        p, r, f = coref_evaluator.get_prf()
        summary_dict["Average F1 (py)"] = f
        print("Average F1 (py): {:.2f}%".format(f * 100))
        summary_dict["Average precision (py)"] = p
        print("Average precision (py): {:.2f}%".format(p * 100))
        summary_dict["Average recall (py)"] = r
        print("Average recall (py): {:.2f}%".format(r * 100))

        muc_p, b_p, ceaf_p = coref_evaluator.get_all_precision()
        muc_r, b_r, ceaf_r = coref_evaluator.get_all_recall()
        muc_f1, b_f1, ceaf_f1 = coref_evaluator.get_all_f1()
        print('\n', "Precision:", "\n")
        print("muc:", "{:.2f}%".format(muc_p * 100), '\n')
        print("b_cube:", "{:.2f}%".format(b_p * 100), '\n')
        print("ceaf:", "{:.2f}%".format(ceaf_p * 100), '\n')

        print('\n', "Recall:", "\n")
        print("muc:", "{:.2f}%".format(muc_r * 100), '\n')
        print("b_cube:", "{:.2f}%".format(b_r * 100), '\n')
        print("ceaf:", "{:.2f}%".format(ceaf_r * 100), '\n')

        print('\n', "F1:", "\n")
        print("muc:", "{:.2f}%".format(muc_f1 * 100), '\n')
        print("b_cube:", "{:.2f}%".format(b_f1 * 100), '\n')
        print("ceaf:", "{:.2f}%".format(ceaf_f1 * 100), '\n')

        return util.make_summary(summary_dict), f

    # def evaluate_neuralcoref(self, session, official_stdout=False):
    #   self.load_eval_data()
    #   coref_predictions = {}
    #   coref_evaluator = metrics.CorefEvaluator()
    #   import re
    #   fuhao = re.compile(r'[\,|\.|\?|\!|\']')
    #   for example_num, (tensorized_example, example) in enumerate(self.eval_data):
    #     # _, _, _, _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
    #     # feed_dict = {i:t for i,t in zip(self.input_tensors, tensorized_example)}
    #     # candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(self.predictions, feed_dict=feed_dict)
    #     ss1 = sum(example["sentences"], [])
    #     ss = ''
    #     for idx, i in enumerate(ss1):
    #       if fuhao.match(i) or idx == 0:
    #         ss = ss + i
    #       elif idx != 0:
    #         ss = ss + ' ' + i
    #     doc = nlp(ss)
    #     # predicted_antecedents = self.get_predicted_antecedents(top_antecedents, top_antecedent_scores)
    #     if not doc._.has_coref:
    #       coref_predictions[example["doc_key"]] = []
    #       continue
    #     # sample : [((16, 16), (19, 23)), ((25, 27), (42, 44), (57, 59)), ((65, 66), (82, 83), (101, 102))]
    #     predictions = []
    #     top_span_starts = []
    #     top_span_ends = []
    #     lookup = {}
    #     conll_token_index = 0
    #     conll_ci = 0
    #     spacy_ci = 0
    #     print()
    #     print('ss', ss)
    #     for i in range(len(doc)):
    #       st = doc[i].text
    #       print(st)
    #       spacy_ci += len(st)
    #       while conll_ci < spacy_ci:
    #         conll_ci += len(ss1[conll_token_index])
    #         conll_token_index += 1
    #       lookup[i] = conll_token_index - 1
    #       print(lookup)
    #     for cluster in doc._.coref_clusters:
    #       _tmp = []
    #       print('cluster:', cluster)
    #       for mention in cluster:
    #         print('mention:', mention)
    #         print('start:', mention.start)
    #         print('end:', mention.end)
    #         print(ss[mention.start:mention.end])
    #         print('look:', ss[lookup[mention.start]:lookup[mention.end]])
    #         # print()
    #         _tmp.append((lookup[mention.start], lookup[mention.end - 1]))
    #         top_span_starts.append(mention.start)
    #         top_span_ends.append(mention.end - 1)
    #       predictions.append(tuple(_tmp))
    #       # print(predictions)
    #     coref_predictions[example["doc_key"]] = predictions
    #     # print(coref_predictions)
    #     if example_num % 10 == 0:
    #       print("Evaluated {}/{} examples.".format(example_num + 1, len(self.eval_data)))
    #
    #   summary_dict = {}
    #   conll_results = conll.evaluate_conll(self.config["conll_eval_path"], coref_predictions, official_stdout)
    #   average_f1 = sum(results["f"] for results in conll_results.values()) / len(conll_results)
    #   summary_dict["Average F1 (conll)"] = average_f1
    #   print("Average F1 (conll): {:.2f}%".format(average_f1))
    #
    #   p, r, f = coref_evaluator.get_prf()
    #   summary_dict["Average F1 (py)"] = f
    #   print("Average F1 (py): {:.2f}%".format(f * 100))
    #   summary_dict["Average precision (py)"] = p
    #   print("Average precision (py): {:.2f}%".format(p * 100))
    #   summary_dict["Average recall (py)"] = r
    #   print("Average recall (py): {:.2f}%".format(r * 100))
    #
    #   return util.make_summary(summary_dict), average_f1
import tensorflow  as tf
import math


def bucket_distance(distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    if want to increase distances. Changing the tf.clip_by_value(combined_idx, 0, *), the * is a border.
    if you want increase 65 ~ 128, the * should get para: 11.
    """
    logspace_idx = tf.to_int32(tf.floor(tf.log(tf.to_float(distances)) / math.log(2))) + 3
    use_identity = tf.to_int32(distances <= 4)
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return tf.clip_by_value(combined_idx, 0, 12)          # 12: 256+


if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(bucket_distance(65)))
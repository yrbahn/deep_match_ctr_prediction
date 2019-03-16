#!/usr/bin/python
# -*- coding: utf-8 -*

import os
import argparse
import glob
import apply_bpe
import codecs
from model import DeepWordMatchModel
from vocab import Vocab

import tensorflow as tf
from tensorflow.contrib import learn
#from konlpy.tag import Kkma
#from nltk.tokenize import sent_tokenize

def args_parser():
    """args parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir',
                        type=str,
                        help='train data directory',
                        required=True)
    parser.add_argument('--eval_data_dir',
                        type=str,
                        help='eval data directory',
                        required=True)
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size',
                        default=1)
    parser.add_argument('--conv_word_size',
                        type=int,
                        default=64)
    parser.add_argument('--fc_output_size',
                        type=int,
                        help="output_size of fully connected layers", 
                        default=256)
    parser.add_argument('--embedding_size',
                        type=int,
                        help='embedding size',
                        default=128)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='train data directory',
                        default=1)
    parser.add_argument('--code_file',
                        type=str,
                        help='code file for segmentation in BPE',
                        default=None)
    parser.add_argument('--vocab_file',
                        type=str,
                        help='vacabulary file',
                        required=True)
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001)
    parser.add_argument('--model_dir',
                        type=str,
                        default=None)
    parser.add_argument('--max_query_len',
                        type=int,
                        default=30)
    parser.add_argument('--max_ad_len',
                        type=int,
                        default=30)
    parser.add_argument('--steps',
                        type=int,
                        default=None)
    return parser
    
def input_fn(data_dir, batch_size, max_query_len,  max_ad_len, num_epochs, bpe, voca, shuffle=True):
    """input function"""

    def _input_fn():
        def _transform_ids(sentence):
            sent = sentence.decode('utf8')
            
            #sentence tokenizer
            _sent = sent.split('\t')

            if len(_sent) != 3:
                raise ValueError("invalid format")

            query = _sent[0]
            ad = _sent[1]
            label = int(_sent[2])
            if bpe != None:
                query = list(map(bpe.segment, query))[:max_query_len]
                ad = list(map(bpe.segment, ad))[:max_ad_len]
            
            query = query + ["PAD"]*(max_query_len-len(query))
            ad = ad + ["PAD"]*(max_ad_len-len(ad))
            
            query_ids = voca.get_ids(query)
            ad_ids = voca.get_ids(ad)

            return query_ids, ad_ids, label

        dataset = tf.contrib.data.TextLineDataset(
            glob.glob(os.path.join(data_dir, "*")))

        dataset = dataset.map(lambda stmt:
                              tuple(tf.py_func(_transform_ids, [stmt], [tf.int64,tf.int64,tf.int64])))
        dataset = dataset.map(lambda query_ids, ad_ids, label:
                              ({"query":query_ids, "ad":ad_ids, }, label))

        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epochs)

        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return features, labels
    return _input_fn

def main(argv=None):
    """main"""    
    args, _ = args_parser().parse_known_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    #check bpe code
    if args.code_file:
        codes = codecs.open(args.code_file, encoding='utf-8')
        bpe = apply_bpe.BPE(codes)
    else:
        bpe = None

    #load vocab
    vocab = Vocab(args.vocab_file)
    
    #train input function for an estimator
    train_input_fn = input_fn(args.train_data_dir,
                              args.batch_size,
                              args.max_query_len,
                              args.max_ad_len,
                              args.num_epochs,
                              bpe,
                              vocab)
    
    deep_match_model = DeepWordMatchModel(vocab.get_size())
    model_fn = deep_match_model.create_model_fn()
    
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=args.model_dir,
                                       params=args)

    # train
    estimator.train(input_fn=train_input_fn,
                    steps=args.steps)


    """
    features = train_input_fn()
    
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(features))
    """
if __name__ == "__main__":
    main()

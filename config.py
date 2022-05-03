# -*- coding: utf-8 -*-


import os

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')


MIN_FREQ = 5
VOCAB_SIZE = 10000
UNK = '<UNK>'
PAD = '<PAD>'

MAX_SEQ_LENGTH = 30
BATCH_SIZE = 64

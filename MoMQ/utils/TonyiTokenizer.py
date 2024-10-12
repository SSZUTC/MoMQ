# -*- coding: utf-8 -*-
# @Time     :  17:05
# @File     : TonyiTokenizer.py
# @Author   : zhencang
# @Team     : XGeneration
import os
import argparse


def get_qwentokenizer():
    from utils.qwentokenizer import build_tokenizer as build_qwentokenizer

    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)
    parser.add_argument('--vocab-file', type=str,
                        default=os.path.join(os.path.dirname(__file__), "qwentokenizer/qwen_15w.tiktoken"),
                        help='Path to the vocab file.')
    parser.add_argument('--merge-file', type=str,
                        default=os.path.join(os.path.dirname(__file__), "qwentokenizer/qwen_15w.tiktoken"),
                        help='Path to the BPE merge file.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0,
                        help='Number of additional vocabulary tokens. '
                             'They are used for span masking in the T5 model')
    parser.add_argument('--tokenizer-type', type=str,
                        default="QWenTokenizer",
                        choices=['BertWordPieceLowerCase',
                                 'BertWordPieceCase',
                                 'GPT2BPETokenizer',
                                 'GPT2ZHBPETokenizer',
                                 'QWenTokenizer'],
                        help='What type of tokenizer to use.')
    parser.add_argument('--tokenizer-model', type=str, default=None,
                        help='Sentencepiece tokenizer model.')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128)
    parser.add_argument('--tensor_model_parallel_size', type=int, default=1)
    parser.add_argument('--add-more-sp-tokens', type=bool, default=True,
                        help='Add more special tokens in vocabulary (only used in QWenTokenizer)')
    # parser.add_argument('--add-more-sp-tokens', action='store_true',
    #                    help='Add more special tokens in vocabulary (only used in QWenTokenizer)')

    args = parser.parse_args()
    print(args.add_more_sp_tokens)

    tokenizer = build_qwentokenizer(args)

    return tokenizer

if __name__ == '__main__':
    tokenizer = get_qwentokenizer()
    print(tokenizer.tokenize("hello world!"))
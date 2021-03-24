import argparse
import sys

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from transformers import BertTokenizerFast
from tqdm import tqdm

from linguistic_style_transfer_model.config import global_config
from linguistic_style_transfer_model.utils import log_initializer

logger = None


def train_word2vec_model(text_file_path, model_file_path):
    # define training data
    # train model
    logger.info("Loading input file and training mode ...")
    model = Word2Vec(sentences=LineSentence(text_file_path), min_count=1, size=global_config.embedding_size)
    # summarize the loaded model
    logger.info("Model Details: {}".format(model))
    # save model
    # model.wv.save_word2vec_format(model_file_path, binary=True)
    model.wv.save_word2vec_format(model_file_path, binary=False)
    logger.info("Model saved")

def train_word2vec_model_zh(text_file_path, model_file_path):
    # define training data
    # train model
    logger.info("Loading input file and training mode ...")

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

    lines = open(text_file_path, 'rt').read().strip().split('\n')
    docs = [tokenizer.tokenize(line) for line in tqdm(lines)]
    
    model = Word2Vec(docs, min_count=1, size=global_config.embedding_size)
    # summarize the loaded model
    logger.info("Model Details: {}".format(model))
    # save model
    # model.wv.save_word2vec_format(model_file_path, binary=True)
    model.wv.save_word2vec_format(model_file_path, binary=False)
    logger.info("Model saved")


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-file-path", type=str, required=True)
    parser.add_argument("--model-file-path", type=str, required=True)
    parser.add_argument("--logging-level", type=str, default="INFO")
    parser.add_argument("--zh", action='store_true', help='Corpus in Chinese')

    options = vars(parser.parse_args(args=argv))
    global logger
    logger = log_initializer.setup_custom_logger(global_config.logger_name, options['logging_level'])

    if options['zh']:
        train_word2vec_model_zh(options['text_file_path'], options['model_file_path'])
    else:
        train_word2vec_model(options['text_file_path'], options['model_file_path'])

    logger.info("Training Complete!")


if __name__ == "__main__":
    main(sys.argv[1:])

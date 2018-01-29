import torch
from torch.autograd import Variable

#
# Functions used to prepare data
#

PAD_token = 0
SOS_token = 1
EOS_token = 2

use_cuda = torch.cuda.is_available()


def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def category_from_string(category_string):
    categories = []
    for cat in category_string:
        categories.append(int(cat))
    return categories


def indexes_from_sentence(vocabulary, sentence):
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def variable_from_sentence(vocabulary, sentence):
    indexes = indexes_from_sentence(vocabulary, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(article, title, vocabulary):
    input_variable = variable_from_sentence(vocabulary, article)
    target_variable = variable_from_sentence(vocabulary, title)
    return input_variable, target_variable


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")

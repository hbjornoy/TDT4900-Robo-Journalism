import sys
import os
from pathlib import Path

dirname = os.path.dirname(os.path.abspath(__file__))
p = Path(dirname)
twolevelsup = str(p.parent.parent)
if twolevelsup not in sys.path:
    sys.path.append(twolevelsup)  # ugly dirtyfix for imports to work

from models.seq2seq.decoder import *
from models.seq2seq.encoder import *
from preprocess.preprocess_pointer import *
from evaluation.seq2seq.evaluate import *


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()

    filename = "cnn_beam_output_rougetest_3.log"
    init_logger(filename)

    if use_cuda:
        if len(sys.argv) < 2:
            log_message("Expected 1 argument: [0] = GPU (0 or 1)")
            exit()
        device_number = int(sys.argv[1])
        if device_number > -1:
            torch.cuda.set_device(device_number)
            log_message("Using GPU: %s" % sys.argv[1])
        else:
            log_message("Not setting specific GPU")

    relative_path = "../data/cnn_pickled/cnn_pointer_50k"
    # relative_path = "../../data/ntb_pickled/ntb_pointer_30k"
    hidden_size = 128
    embedding_size = 100
    n_layers = 1
    dropout_p = 0.0
    # load_file = "../../models/GAN_trained_models/epoch3_cnn_generator_rougetest_1.pth.tar"
    # load_file = "../../models/pretrained_models/after_gan/epoch1_cnn_generator_unk_fix_rl_metricl.pth.tar"
    #load_file = "models/pretrained_models/after_gan/epoch1_cnn_generator_GAN_test.pth.tar"
    # load_file = "../../models/pretrained_models/after_gan/ntb_generator_test_save_2.tar"
    # HB
    load_file = "models/pretrained_models/cnn/epoch13_cnn_test1.pth.tar"

    summary_pairs, vocabulary = load_dataset(relative_path)
    encoder = EncoderRNN(vocabulary.n_words, embedding_size, hidden_size, n_layers=n_layers)

    max_article_length = max(len(pair.article_tokens) for pair in summary_pairs) + 1
    max_abstract_length = max(len(pair.abstract_tokens) for pair in summary_pairs) + 1

    decoder = PointerGeneratorDecoder(hidden_size, embedding_size, vocabulary.n_words, max_length=max_article_length,
                                      n_layers=n_layers, dropout_p=dropout_p)

    try:
        model_state_encoder, model_state_decoder = load_state(load_file)
        encoder.load_state_dict(model_state_encoder)
        decoder.load_state_dict(model_state_decoder)
    except FileNotFoundError as e:
        log_error_message("No file found: exiting " + str(e))
        exit()

    log_message("Done loading the model")

    encoder.eval()
    decoder.eval()

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    summary_pairs = summary_pairs[-13000:-11990]
    log_message("Evaluating %d examples" % len(summary_pairs))

    config = {}
    config['evaluate'] = {}
    config['evaluate']['expansions'] = 3
    config['evaluate']['keep_beams'] = 20
    config['evaluate']['return_beams'] = 3

    evaluate(config, summary_pairs, vocabulary, encoder, decoder, max_article_length, print_status=True)

    log_message("Done")
    print("Done", flush=True)

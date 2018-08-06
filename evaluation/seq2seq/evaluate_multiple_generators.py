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
#from evaluation.seq2seq.calculate_rouge_in_folder import read_directory


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return state['model_state_encoder'], state['model_state_decoder']
    else:
        raise FileNotFoundError


def read_directory(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".tar"):
            yield os.path.join(directory, file)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    models_directory = sys.argv[1]
    files = list(read_directory(models_directory))
    files.sort()
    output_directory = twolevelsup + "/" + "/".join(models_directory.split("/")[0:-1]) + "/models_first_eval/"
    print("Number of files to run: %d" % len(files), flush=True)

    for i in range(0, len(files)):
        load_file = files[i]
        filename = output_directory + load_file.split('/')[-1] + "_eval.log"
        # clear logfile first
        with open(filename, 'w'):
            pass
        # create new logger for each model
        logger = setup_logger(filename, filename)

        if use_cuda:
            if len(sys.argv) < 3:
                logger.info("Expected 1 argument: [0] = GPU (0 or 1)")
                exit()
            device_number = int(sys.argv[2])
            if device_number > -1:
                torch.cuda.set_device(device_number)
                logger.info("Using GPU: %s" % sys.argv[2])
            else:
                logger.info("Not setting specific GPU")

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
            logger.info("No file found: exiting " + str(e))
            exit()
        logger.info("Done loading the model")

        encoder.eval()
        decoder.eval()

        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

        summary_pairs = summary_pairs[-13000:-11990]
        logger.info("Evaluating %d examples" % len(summary_pairs))

        config = {}
        config['evaluate'] = {}
        config['evaluate']['expansions'] = 3
        config['evaluate']['keep_beams'] = 20
        config['evaluate']['return_beams'] = 3

        evaluate(config, summary_pairs, vocabulary, encoder, decoder, max_article_length, logger, print_status=True)
        logger.info("Done")
    print("Done", flush=True)

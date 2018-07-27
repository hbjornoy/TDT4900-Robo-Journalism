import os
import sys
import time
from tensorboardX import SummaryWriter
import re

from pathlib import Path
dirname = os.path.dirname(os.path.abspath(__file__))
p = Path(dirname)
twolevelsup = str(p.parent.parent)
if twolevelsup not in sys.path:
    sys.path.append(twolevelsup)  # ugly dirtyfix for imports to work

from evaluation.seq2seq.calculate_rouge_3 import calculate_rouge_3
from evaluation.seq2seq.split_beamsearch_to_multiple_files import split_beamsearch_to_multiple_files


def read_directory(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".log"):
            yield os.path.join(directory, file)


if __name__ == '__main__':

    # Parameters
    test_data = False
    directory = "output_for_eval/seq2seq/cnn_models_first_eval/"
    
    # validation set
    path_to_reference = "output_for_eval/seq2seq/split_data/reference/"
    path_to_modelsummary = "output_for_eval/seq2seq/split_data/modelsummary/"
    #num_summaries = 2000
    num_summaries = 10 # 1010

    # test set
    if test_data:
        path_to_reference = "../for_rouge/test_data/reference/"
        path_to_modelsummary = "../for_rouge/test_data/modelsummary/"
        num_summaries = 11000

    files = list(read_directory(directory))
    # sort the files properly(numerically) by epoch
    files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    # make summarywriter for logging to tensorboard
    writer = SummaryWriter('log/rouge/new_rouge')

    print("Number of files to run: %d" % len(files), flush=True)

    for i in range(0, len(files)):
        f = files[i]
        print("######################################################", flush=True)
        print("######################################################", flush=True)
        print("Number %d of %d" % (i+1, len(files)))
        print("Evaluating file: %s" % f, flush=True)
        print("Splitting file", flush=True)
        split_beamsearch_to_multiple_files(f, path_to_reference, path_to_modelsummary, num_summaries)
        print("Sleeping for 3 seconds", flush=True)
        time.sleep(3)
        print("Calculating rouge", flush=True)
        output = calculate_rouge_3(path_to_reference, path_to_modelsummary)

        # denote with epoch it is by extracting from filename
        try:
            name_of_file = f.split('/')[-1]
            epoch = re.findall(r'\d+', name_of_file)
            assert len(epoch) == 1
            epoch = int(epoch[0])
        except AssertionError as e:
            print('Expected a format with only one number (ex:epoch14_baseline.pth.tar_eval.log)')
            raise e

        for key, value in output.items():
            # process key name into folders names
            filtered_list = list(filter(lambda x: x != '1.2', key.split('_')))
            metric = '_'.join(filtered_list[0:2])
            metric_detail = '_'.join(filtered_list[2:])

            writer.add_scalar(metric + '/' + metric_detail, value, epoch)
        # return output_dict
        print("Done with file", flush=True)
        print("######################################################", flush=True)
        print("######################################################\n\n\n", flush=True)

    print("Done", flush=True)








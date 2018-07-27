import json
import re


def read_file(path):
    text = open(path, encoding='utf-8').read()
    if path.endswith(".log"):
        text = clean_logger_output(text)  # when using logger
    titles = []
    output = []
    not_truth = False
    for line in text.split('\n'):
        if line.startswith('>'):
            if not_truth:
                titles.pop()  # temp fix when we apparently do not get all the output...
            not_truth = True
        elif line.startswith('='):
            titles.append(line[1:])
        elif line.startswith('<') and not_truth:
            output.append(line[1:])
            not_truth = False

    output = clean_modelsummary(output)
    titles = clean_reference(titles)
    return titles, output


def clean_modelsummary(input_txt):
    output_txt = []
    for line in input_txt:
        line = line.split(" ")
        line = " ".join(line[2:])
        # line = line[16:]
        # line = re.sub(r'\d+', '', line)
        # line = line[3:]
        line = re.sub(r'<EOS>', '', line)
        line = re.sub(r'<PAD>', '', line)
        line = line.strip()
        output_txt.append(line)
    return output_txt


def clean_reference(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'<EOS>', '', line)
        line = re.sub(r'<PAD>', '', line)
        line = line.strip()
        output_txt.append(line)
    return output_txt


def clean_logger_output(text):
    output = ""
    for line in text.split('\n'):
        cleaned_line = re.sub('^(.*?)INFO - ', '', line)
        cleaned_line = re.sub('^(.*?)ERROR - ', '', cleaned_line)
        if len(cleaned_line) > 0:
            output += cleaned_line + '\n'
    return output


def add_back_delimiter(sentence_list, delimiter):
    for i in range(0, len(sentence_list)-1):
        sentence_list[i] += " " + delimiter
    return sentence_list


def split_sentence(sentence):
    sentences = []
    temp = add_back_delimiter(sentence.split(" . "), ".")
    for k in temp:
        temp2 = add_back_delimiter(k.split(" ? "), "?")
        for j in temp2:
            temp3 = add_back_delimiter(j.split(" ! "), "!")
            for t in temp3:
                if t.startswith("<"):
                    t = t[1:]
                sentences.append(t)
    return "\n".join(sentences).strip()


def split_beamsearch_to_multiple_files(path, path_to_reference, path_to_modelsummary, num_summaries):
    print("Started extracting titles...")
    reference, hypothesis = read_file(path)

    if len(reference) < num_summaries:
        print("Error - not enough references, %d references and %d summaries" % (len(reference), num_summaries))
        exit()
    if len(hypothesis) < num_summaries:
        print("Error - not enough hypotheses, %d hypothesis and %d summaries" % (len(hypothesis), num_summaries))
        exit()

    reference = reference[:num_summaries]
    hypothesis = hypothesis[:num_summaries]

    if len(reference) != len(hypothesis):
        print("Error - not equal amount of references and hypotheses")
        print("%d references and %d hypothesis" % (len(reference), len(hypothesis)))
        exit()

    for i in range(0, len(reference)):
        reference[i] = split_sentence(reference[i])

    for i in range(0, len(hypothesis)):
        hypothesis[i] = split_sentence(hypothesis[i])

    for i in range(0, len(reference)):
        with open(path_to_reference + "%d_reference.txt" % i, 'w', encoding='utf-8') as file:
            file.write(reference[i])

    for i in range(0, len(hypothesis)):
        with open(path_to_modelsummary + "%d_modelsummary.txt" % i, 'w', encoding='utf-8') as file:
            file.write(hypothesis[i])


if __name__ == '__main__':

    # path = '../output_for_eval/old1/cnn_beam_output_2_13epoch_3_20_1000.log'

    # path = '../output_for_eval/cnn_beam_output_epoch16_extratrain_2.log'
    # path = '../output_for_eval/cnn_beam_output_2_8epoch.log'
    # path = '../output_for_eval/cnn_beam_gan_long_lr0001.log'
    # path = '../output_for_eval/old_rouge/output_eval_rouge_argmax_trigram_metric_pretrained.txt'
    # path = '../output_for_eval/beam_output_rouge_test.txt'
    path = '../output_for_eval/beam_output_seqGAN_strat_rouge_1_epoch1_2quarter.txt'
    # path = '../output_for_eval/old1/cnn_pretrained_1.log'

    path_to_reference = "../for_rouge/pretrained1/reference_new/"
    path_to_modelsummary = "../for_rouge/pretrained1/cnn_pretrain_new/"

    num_summaries = 2000

    split_beamsearch_to_multiple_files(path, path_to_reference, path_to_modelsummary, num_summaries)

    # len_dict = {}
    # max_length = 0
    # for hyp in hypothesis:
    #     current_length = len(hyp.split(" "))
    #     # if current_length > max_length:
    #     #     max_length = current_length
    #     if current_length in len_dict.keys():
    #         len_dict[current_length] += 1
    #     else:
    #         len_dict[current_length] = 1
    # print("Max length")
    # print(json.dumps(len_dict, indent=2))
    # print("Done")

    print("Done")

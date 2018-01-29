import random

from torch import nn

from evaluation.seq2seq.evaluate import *
from utils.batching import *
from utils.data_prep import *
from utils.time_utils import *


# Train one batch
def train(config, input_variable, input_lengths, target_variable, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    batch_size = config['train']['batch_size']
    teacher_forcing_ratio = config['train']['teacher_forcing_ratio']

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_target_length = max(target_lengths)
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)

    encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
    num_layers = config['model']['n_layers']
    encoder_hidden = encoder_hidden.repeat(num_layers, 1, 1)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        batch_size)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs,
                                                                        batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
            loss += criterion(decoder_output, target_variable[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def train_iters(config, articles, titles, eval_articles, eval_titles, vocabulary, encoder, decoder, max_length,
                encoder_optimizer, decoder_optimizer, writer, start_epoch=1, total_runtime=0, with_categories=False):

    start = time.time()
    print_loss_total = 0  # Reset every print_every
    lowest_loss = 999  # TODO: FIX THIS. save and load

    n_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    print_every = config['log']['print_every']

    criterion = nn.NLLLoss()

    num_batches = int(len(articles) / batch_size)
    n_iters = num_batches * n_epochs

    print("Starting training", flush=True)
    for epoch in range(start_epoch, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)
        batch_loss_avg = 0

        # shuffle articles and titles (equally)
        c = list(zip(articles, titles))
        random.shuffle(c)
        articles_shuffled, titles_shuffled = zip(*c)

        # split into batches
        article_batches = list(chunks(articles_shuffled, batch_size))
        title_batches = list(chunks(titles_shuffled, batch_size))

        for batch in range(num_batches):
            categories, input_variable, input_lengths, target_variable, target_lengths = random_batch(batch_size,
                  vocabulary, article_batches[batch], title_batches[batch], max_length, with_categories)

            loss = train(config, input_variable, input_lengths, target_variable, target_lengths,
                         encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            print_loss_total += loss
            batch_loss_avg += loss
            # calculate number of batches processed
            itr = (epoch-1) * num_batches + batch + 1

            if itr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                progress, total_runtime = time_since(start, itr / n_iters, total_runtime)
                start = time.time()
                print('%s (%d %d%%) %.4f' % (progress, itr, itr / n_iters * 100, print_loss_avg), flush=True)
                if print_loss_avg < lowest_loss:
                    lowest_loss = print_loss_avg
                    print(" ^ Lowest loss so far", flush=True)

        # log to tensorboard
        writer.add_scalar('loss', batch_loss_avg / num_batches, epoch)

        # save each epoch
        print("Saving model", flush=True)
        itr = epoch * num_batches
        _, total_runtime = time_since(start, itr / n_iters, total_runtime)
        save_state({
            'epoch': epoch,
            'runtime': total_runtime,
            'model_state_encoder': encoder.state_dict(),
            'model_state_decoder': decoder.state_dict(),
            'optimizer_state_encoder': encoder_optimizer.state_dict(),
            'optimizer_state_decoder': decoder_optimizer.state_dict()
        }, config['experiment_path'] + "/" + config['save']['save_file'])

        encoder.eval()
        decoder.eval()
        calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_length,
                                   eval_articles, eval_titles)
        encoder.train()
        decoder.train()


def save_state(state, filename):
    torch.save(state, filename)

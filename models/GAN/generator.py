import random

from utils.data_prep import *
from utils.logger import *
import time


class Generator:
    def __init__(self, vocabulary, encoder, decoder, encoder_optimizer, decoder_optimizer, mle_criterion,
                 policy_criterion, batch_size, use_cuda, beta, num_monte_carlo_samples):
        self.vocabulary = vocabulary
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.mle_criterion = mle_criterion
        self.policy_criterion = policy_criterion
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.beta = beta
        self.num_monte_carlo_samples = num_monte_carlo_samples
        self.updates = 0
        self.cumulative_reward = 0.0
        self.allow_negative_rewards = False

    # discriminator is used to calculate reward
    # target batch is used for MLE
    def train_on_batch(self, input_variable_batch, full_input_variable_batch, input_lengths, full_target_variable_batch,
                       target_lengths, discriminator, max_monte_carlo_length, target_variable):

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        max_target_length = max(target_lengths)

        init_encoder_time_start = time.time()
        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        timings[timings_var_init_encoder] += (time.time() - init_encoder_time_start)

        # TEACHER FORCING MLE START
        mle_loss = self.get_teacher_forcing_mle(encoder_hidden, encoder_outputs, max_target_length,
                                                full_input_variable_batch, full_target_variable_batch, target_variable)

        # POLICY ITERATION START
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        # Testing with setting a lower upper limit to monte carlo sequence length
        monte_carlo_length = min(max_target_length, max_monte_carlo_length)

        policy_loss = 0
        total_reward = 0
        adjusted_reward = 0
        full_sequence_rewards = []
        full_policy_values = []

        accumulated_sequence = None

        policy_iteration_time_start = time.time()
        policy_iteration_break_early = False

        sample_rate = 0.33
        num_samples = 0

        # Do policy iteration
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)

            log_output = torch.log(decoder_output.clamp(min=1e-8))

            # TODO: Remove this when we use teacher forcing
            # mle_loss += self.mle_criterion(log_output, full_target_variable_batch[di])

            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

            # Sample things
            # Currently always sampling the first token (to make sure there is at least 1 sampling per batch)
            sampling = True if random.random() < sample_rate else False
            if sampling or di == 0:
                num_samples += 1
                sampled_token = decoder_output.data.multinomial(1)
                for token_index in range(0, len(sampled_token)):
                    if sampled_token[token_index][0] >= self.vocabulary.n_words:
                        sampled_token[token_index][0] = UNK_token
                monte_carlo_input = Variable(sampled_token)

                # calculate policy value
                policy_target = Variable(sampled_token.squeeze(1))
                current_policy_value = self.policy_criterion(log_output, policy_target)
                full_policy_values.append(current_policy_value)
                # calculate policy loss using monte carlo search
                accumulated_reward = 0

                monte_carlo_time_start = time.time()
                sample = self.monte_carlo_expansion(monte_carlo_input, decoder_hidden, encoder_outputs,
                                                    full_input_variable_batch, accumulated_sequence, monte_carlo_length)
                monte_carlo_outer_time_start = time.time()
                current_reward = discriminator.evaluate(sample)
                accumulated_reward += current_reward
                # add cumulative reward to calculate running average baseline
                self.cumulative_reward += current_reward.mean().data[0]
                self.updates += 1
                timings[timings_var_monte_carlo_outer] += (time.time() - monte_carlo_outer_time_start)
                timings[timings_var_monte_carlo] += (time.time() - monte_carlo_time_start)

                reward = accumulated_reward / self.num_monte_carlo_samples
                full_sequence_rewards.append(reward)
                total_reward += reward  # used for printing only

            # Update accumulated sequence
            if accumulated_sequence is None:
                accumulated_sequence = decoder_input
            else:
                accumulated_sequence = torch.cat((accumulated_sequence, decoder_input), 1)

            # Break the policy iteration loop if all the variables in the batch is at EOS or PAD
            if is_whole_batch_pad_or_eos(ni):
                decode_breakings[decode_breaking_policy] += di
                policy_iteration_break_early = True
                break

        if not policy_iteration_break_early:
            decode_breakings[decode_breaking_policy] += max_target_length - 1

        # Calculate running average baseline
        avg = self.cumulative_reward / self.updates
        baseline = Variable(torch.cuda.FloatTensor([avg]))

        # Print baseline value for testing purposes
        # log_message("Baseline value: %.6f" % baseline.data[0])

        for i in range(0, len(full_sequence_rewards)):
            temp_adjusted_reward = full_sequence_rewards[i] - baseline
            if not self.allow_negative_rewards:
                for j in range(0, len(temp_adjusted_reward.data)):
                    if temp_adjusted_reward.data[j] < 0.0:
                        temp_adjusted_reward.data[j] = 0.0
            adjusted_reward += temp_adjusted_reward
            current_policy_loss = temp_adjusted_reward * full_policy_values[i]
            reduced_policy_loss = current_policy_loss.mean()
            policy_loss += reduced_policy_loss

        # Test to look at the accumulated sequence
        # multinomial_1 = accumulated_sequence[0].data
        # log_message("Last one")
        # log_message(get_sentence_from_tokens_unked(multinomial_1, self.vocabulary))
        # exit()

        total_loss = self.beta * policy_loss + (1 - self.beta) * mle_loss

        timings[timings_var_policy_iteration] += (time.time() - policy_iteration_time_start)

        backprop_time_start = time.time()

        total_loss.backward()

        clip = 2
        torch.nn.utils.clip_grad_norm(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        timings[timings_var_backprop] += (time.time() - backprop_time_start)

        total_reward = (total_reward / num_samples).mean()
        adjusted_reward = (adjusted_reward / num_samples).mean()

        return total_loss.data[0], mle_loss.data[0], policy_loss.data[0], total_reward, adjusted_reward

    def get_teacher_forcing_mle(self, encoder_hidden, encoder_outputs, max_target_length, full_input_variable_batch,
                                full_target_variable_batch, target_variable):
        mle_loss = 0
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            log_output = torch.log(decoder_output.clamp(min=1e-8))
            mle_loss += self.mle_criterion(log_output, full_target_variable_batch[di])
            decoder_input = target_variable[di]
        return mle_loss

    def monte_carlo_expansion(self, sampled_token, decoder_hidden, encoder_outputs, full_input_variable_batch,
                              initial_sequence, max_sample_length):

        # TODO: Do we need to set eval mode here? Guess not?
        # Should we set require_grad = False? (Seems not)
        # for param in self.decoder.parameters():
        #     param.require_grad = False

        start_check_for_pad_and_eos = int(max_sample_length / 3) * 2

        if initial_sequence is not None:
            start = len(initial_sequence.data[0]) + 1
            decoder_output_variables = torch.cat((initial_sequence, sampled_token), 1)
        else:
            start = 1
            decoder_output_variables = sampled_token

        decoder_input = sampled_token

        monte_carlo_sampling_break_early = False
        for di in range(start, max_sample_length):
            monte_carlo_inner_time_start = time.time()
            decoder_output, decoder_hidden, _ \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_monte_carlo_inner] += (time.time() - monte_carlo_inner_time_start)

            before_topk_monte = time.time()
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)
            timings[timings_var_monte_carlo_top1] += (time.time() - before_topk_monte)

            monte_carlo_cat_time_start = time.time()
            decoder_output_variables = torch.cat((decoder_output_variables, decoder_input), 1)
            timings[timings_var_monte_carlo_cat] += (time.time() - monte_carlo_cat_time_start)

            if di > start_check_for_pad_and_eos:
                if is_whole_batch_pad_or_eos(ni):
                    monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += di
                    monte_carlo_sampling[monte_carlo_sampling_num] += 1
                    monte_carlo_sampling_break_early = True
                    break

        if not monte_carlo_sampling_break_early:
            monte_carlo_sampling[decode_breaking_monte_carlo_sampling] += max_sample_length - 1
            monte_carlo_sampling[monte_carlo_sampling_num] += 1

        return decoder_output_variables

    # Used to create fake data samples to train the discriminator
    # Returned values as batched sentences as variables
    def create_samples(self, input_variable_batch, full_input_variable_batch, input_lengths, max_sample_length,
                       pad_length):

        self.encoder.eval()
        self.decoder.eval()

        encoder_outputs, encoder_hidden = self.encoder(input_variable_batch, input_lengths, None)
        encoder_hidden = concat_encoder_hidden_directions(encoder_hidden)
        # Multiple layers are currently removed for simplicity
        decoder_input = Variable(torch.LongTensor([SOS_token] * self.batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input
        decoder_hidden = encoder_hidden

        decoder_outputs = [[] for _ in range(0, self.batch_size)]
        create_fake_sample_break_early = False

        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_sample_length):

            create_fake_inner_time_start = time.time()
            decoder_output, decoder_hidden, decoder_attention \
                = self.decoder(decoder_input, decoder_hidden, encoder_outputs, full_input_variable_batch,
                               self.batch_size)
            timings[timings_var_create_fake_inner] += (time.time() - create_fake_inner_time_start)

            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            for token_index in range(0, len(ni)):
                if ni[token_index][0] >= self.vocabulary.n_words:
                    ni[token_index][0] = UNK_token
            decoder_input = Variable(ni)

            decoder_output_data = ni.cpu().numpy()
            for batch_index in range(0, len(decoder_output_data)):
                decoder_outputs[batch_index].append(decoder_output_data[batch_index].item())

            if is_whole_batch_pad_or_eos(ni):
                decode_breakings[decode_breaking_fake_sampling] += di
                create_fake_sample_break_early = True
                break

        if not create_fake_sample_break_early:
            decode_breakings[decode_breaking_fake_sampling] += max_sample_length - 1

        decoder_outputs_padded = [pad_seq(s, pad_length) for s in decoder_outputs]
        decoder_output_variables = Variable(torch.LongTensor(decoder_outputs_padded))
        if self.use_cuda:
            decoder_output_variables = decoder_output_variables.cuda()

        self.encoder.train()
        self.decoder.train()

        return decoder_output_variables

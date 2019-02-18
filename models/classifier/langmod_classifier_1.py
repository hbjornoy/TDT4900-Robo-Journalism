import torch
import torch.nn as nn
import torch.nn.functional as func


class LMDiscriminator(nn.Module):
	def __init__(self, vocab_size, hidden_size, num_kernels, kernel_sizes, dropout_p):
		# define model structure and layers
		"""
		static LM
		language model should be treated as bias?
		pass through, but not train the LM
		the importance given to LM can be trained. HOW?

		will the input be encoded? must get code to translate for LM
		"""



	def forward(self, x):
		#define forward

import torch

class TextDecoder:
    """Base class to be inherited by all decoding strategies. Contains
    implementations that are common for all strategies.

    Args:
        vocab (list): Collection of all words in vocabulary.

    """

    def __init__(self, vocab):
        self._vocab = vocab
        self._vocab_size = vocab.get_size()

        # Lists to store completed sequences and scores
        self._complete_seqs = []
        self._complete_seqs_scores = []

    def init_batch(self, sample_list):
        img_size = sample_list.image_feature_0.size()
        self._batch_size, feature_size_1, feature_size_2 = img_size
        t_batch_size = self._batch_size * self._decode_size
        self.seqs = sample_list.answers.new_full(
            (t_batch_size, 1), self._vocab.SOS_INDEX, dtype=torch.long
        )
        sample_list.image_feature_0 = (
            sample_list.image_feature_0.unsqueeze(1)
            .expand(-1, self._decode_size, -1, -1)
            .reshape(t_batch_size, feature_size_1, feature_size_2)
        )
        self.sample_list = sample_list
        return sample_list

    def add_next_word(self, seqs, prev_word_inds, next_word_inds):
        return torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

    def find_complete_inds(self, next_word_inds):
        incomplete_inds = []
        for ind, next_word in enumerate(next_word_inds):
            if next_word != self._vocab.EOS_INDEX:
                incomplete_inds.append(ind)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
        return complete_inds, incomplete_inds

    def update_data(self, data, prev_word_inds, next_word_inds, incomplete_inds):
        data["texts"] = next_word_inds[incomplete_inds].unsqueeze(1)
        h1 = data["state"]["td_hidden"][0][prev_word_inds[incomplete_inds]]
        c1 = data["state"]["td_hidden"][1][prev_word_inds[incomplete_inds]]
        h2 = data["state"]["lm_hidden"][0][prev_word_inds[incomplete_inds]]
        c2 = data["state"]["lm_hidden"][1][prev_word_inds[incomplete_inds]]
        data["state"] = {"td_hidden": (h1, c1), "lm_hidden": (h2, c2)}
        return data

import numpy as np
import argparse

BLANK_SYMBOL = "_"


def load_mat(path_to_mat):
    """
    Load mat file and return numpy array
    :param path_to_mat: path to mat file
    :return: numpy array
    """
    return np.load(path_to_mat)


def ctc(word: str, get_letter_time_prob: callable, T: int) -> float:
    z = get_valid_z_pattern(word)

    linear_prog = np.ones((len(z), T)) * -1  # S x T

    def calc_prob(s: int, t: int):
        if s < 0 or t < 0:
            return 0
        if t == 0:
            if s > 1:
                linear_prog[s][t] = 0
            else:
                linear_prog[s][t] = get_letter_time_prob(t, z[s])
        if linear_prog[s][t] == -1:
            if z[s] == BLANK_SYMBOL or (s > 1 and z[s] == z[s - 2]):
                linear_prog[s][t] = (calc_prob(s - 1, t - 1) + calc_prob(s, t - 1)) * get_letter_time_prob(t, z[s])
            else:
                linear_prog[s][t] = (calc_prob(s - 2, t - 1) + calc_prob(s - 1, t - 1) + calc_prob(s, t - 1)) * get_letter_time_prob(t, z[s])
        return linear_prog[s][t]

    final = calc_prob(len(z) - 1, T - 1) + calc_prob(len(z) - 2, T - 1)
    return final


def get_valid_z_pattern(word):
    z = [BLANK_SYMBOL]
    for c in word:
        z.append(c)
        z.append(BLANK_SYMBOL)
    return z


def get_prob_from_time_letter_func(probs_mat: np.array, vocabulary: str) -> callable:
    assert probs_mat.shape[1] == len(vocabulary)

    def prob_from_time_letter(time: int, letter: str):
        letter_index = vocabulary.index(letter)
        return probs_mat[time][letter_index]

    return prob_from_time_letter


def parse_args():
    """
    Parse arguments
    :return: args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_mat", type=str, help="path to mat file")
    parser.add_argument("word", type=str, help="word to parse")
    parser.add_argument("vocabulary", type=str, help="string for vocabulary (no blanks)")

    return parser.parse_args()


def main(mat, word, vocabulary):
    vocab = BLANK_SYMBOL + vocabulary
    prob_from_time_letter = get_prob_from_time_letter_func(mat, vocab)
    T, V = mat.shape
    word_prob = ctc(word, prob_from_time_letter, T)
    print("%.3f" % word_prob)
    return word_prob


if __name__ == "__main__":
    args = parse_args()
    mat = load_mat(args.path_to_mat)
    main(mat, args.word, args.vocabulary)

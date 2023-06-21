import numpy as np
from ctc_decoder import probability
from ex4 import main

abc = "abcdefghijklmnopqrstuvwxyz"


def run_fixed_probability(probs_mat, tested_word, vocab):
    new_mat = np.empty_like(probs_mat)
    new_mat[:, 0:-1] = probs_mat[:, 1:]
    new_mat[:, -1] = probs_mat[:, 0]
    return probability(new_mat, tested_word, vocab)


def test_simple():
    vocab = "abc"
    T = 5
    tested_word = "abcba"
    probs_mat = np.ones((T, len(vocab) + 1)) * 0.25
    our_probability = main(probs_mat, tested_word, vocab)
    expected_prob = run_fixed_probability(probs_mat, tested_word, vocab)
    assert our_probability == expected_prob


def test_simple3():
    vocab = "abc"
    tested_word = "ab"
    probs_mat = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.1],
            [0.3, 0.4, 0.1, 0.2],
            [0.4, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    our_probability = main(probs_mat, tested_word, vocab)
    expected_prob = run_fixed_probability(probs_mat, tested_word, vocab)
    assert our_probability == expected_prob


def test_simple2():
    probs_mat = np.array(
        [
            [0.01933756, 0.51991086, 0.46075158],
            [0.57304014, 0.20393454, 0.22302532],
            [0.28330213, 0.67954595, 0.03715192],
            [0.65280397, 0.20348597, 0.14371005],
            [0.39299785, 0.4588467, 0.14815545],
            [0.19644894, 0.06278048, 0.74077058],
            [0.5097096, 0.37550606, 0.11478435],
            [0.30638284, 0.47182778, 0.22178938],
            [0.22013477, 0.39090866, 0.38895658],
        ]
    )
    tested_word = "jjj"
    vocab = "aj"

    while (probs_mat.sum(axis=1) != 1).any():
        print("fixing probs_mat")
        probs_mat /= probs_mat.sum(axis=1, keepdims=True)

    our_probability = main(probs_mat, tested_word, vocab)
    expected_prob = run_fixed_probability(probs_mat, tested_word, vocab)
    assert np.isclose(
        our_probability, expected_prob
    ), f"our_probability={our_probability}, expected_prob={expected_prob}\n probs_mat={probs_mat}\n tested_word={tested_word}\n vocab={vocab}"


def test_mat_file():
    mat_path = "mat.npy"
    probs_mat = np.load(mat_path)
    tested_word = "bb"
    vocab = "ab"
    our_probability = main(probs_mat, tested_word, vocab)
    expected_prob = run_fixed_probability(probs_mat, tested_word, vocab)
    assert np.isclose(
        our_probability, expected_prob
    ), f"our_probability={our_probability}, expected_prob={expected_prob}\n probs_mat={probs_mat}\n tested_word={tested_word}\n vocab={vocab}"


def test_random():
    T = np.random.randint(3, 30)
    vocab = "".join(np.random.choice(list(abc), size=np.random.randint(2, 24), replace=False))
    tested_word = "".join(np.random.choice(list(vocab), size=np.random.randint(2, T), replace=True))

    probs_mat = np.random.dirichlet(np.ones(len(vocab) + 1), size=T)

    our_probability = main(probs_mat, tested_word, vocab)
    expected_prob = run_fixed_probability(probs_mat, tested_word, vocab)
    assert np.isclose(
        our_probability,
        expected_prob,
    ), f"our_probability={our_probability}, expected_prob={expected_prob}\n probs_mat={probs_mat}\n tested_word={tested_word}\n vocab={vocab}"


def test_100_random_tests():
    for i in range(100):
        test_random()
        print(f"test {i} passed")

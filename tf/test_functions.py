from final_sda import get_batch_generator, repeat_generator, merge_generators
from functools import wraps


def test(f):
    """Simple decorator that evaluates a test function."""

    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            result = f(*args, **kwargs)
            print("Successfully passed unit test %s." % f)
            return result
        except AssertionError as e:
            print("Failed to pass unit test %s. Error trace is printed below:" % f)
            print(e)

    return wrapped


@test
def test_get_batch_generator():
    g1 = get_batch_generator("../data/rose/small/smallSAMPart01_test_y_r.csv", 100, skip_header=True)
    count = [1 for _ in g1]
    assert len(count) == 20, "Failed simple batch count test"

    g2 = get_batch_generator("../data/rose/small/smallSAMPart01_test_y_r.csv", 100, skip_header=True, repeat=9)
    count = [1 for _ in g2]
    assert len(count) == 200, "Failed repeat batch count test"

    g3 = get_batch_generator("../data/fake/test.csv", 1, skip_header=True, repeat=9)
    count = [batch for batch in g3]
    assert len(count) == 40, "Failed single unit batch test"
    assert all([count[4 * i] == count[4 * i + 4] for i in range(9)]), "Failed repeat batch equality test"

    g4 = get_batch_generator("../data/fake/test.csv", 2, skip_header=False, repeat=1)
    count = [batch for batch in g4]
    assert len(count) == 6, "Failed edge case cutoff"
    assert len(count[2]) == 1, "Failed edge case cutoff length"

    g5 = get_batch_generator("../data/fake/test.csv", 3, skip_header=False, repeat=2)
    [print(batch) for batch in g5]

    g6 = get_batch_generator("../data/fake/test.csv", 2, skip_header=True, repeat=4)
    [print(batch) for batch in g6]


@test
def test_merge_generators():
    g7 = get_batch_generator("../data/fake/test.csv", 2, skip_header=True, repeat=2)
    g8 = get_batch_generator("../data/fake/test.csv", 3, skip_header=False, repeat=2)
    g9 = merge_generators(g7, g8)
    g7 = get_batch_generator("../data/fake/test.csv", 2, skip_header=True, repeat=2)
    g8 = get_batch_generator("../data/fake/test.csv", 3, skip_header=False, repeat=2)
    for x, y in zip(g7, g8):
        compare = next(g9)
        print(compare)
        assert compare[0] == x and compare[1] == y, "Not merged correctly"


def main():
    test_get_batch_generator()

if __name__ == "__main__":
    main()

from unittest import TestCase
from buffer import ReplayBuffer
from experience import VoidExperience as Experience


class TestReplayBuffer(TestCase):

    def null_experience_list(self, count=100):
        return [Experience(None, None, None, None, None) for _ in range(count)]

    def test_init(self):

        # Correct input
        max_lengths = [1, 1, 5, 10, 1000, 10000, 99999999]
        start_lengths = [0, 1, 4, 9, 970, 20, 867698]
        for start_length, max_length in zip(start_lengths, max_lengths):
            buffer = ReplayBuffer(max_length=max_length, start_length=start_length)
            self.assertEqual(buffer.start_length, start_length)
            self.assertEqual(buffer.max_length, max_length)
            self.assertEqual(buffer.buffer, [])

        error_start_lengths = [1, 5, 10, 1000, 10000, 99999999]
        error_max_lengths = [0, 4, 9, 970, 20, 867698]
        for start_length, max_length in zip(error_start_lengths, error_max_lengths):
            with self.assertRaises(AssertionError):  # can fix lack of message with class extension, but not worth it
                ReplayBuffer(max_length=max_length, start_length=start_length),

        test_list = self.null_experience_list(10)
        test_buffer = ReplayBuffer(100, 10, test_list)
        self.assertEqual(test_buffer.buffer, test_list, "Buffer not created correctly.")
    #, f"Did not raise expected error: max_length: {max_length}, start_length: {start_length}"):

    def test_experience_count(self):
        counts = [1, 5, 10, 20, 1000, 100000, 999999]
        for count in counts:
            buffer = ReplayBuffer(start_length=count//2, max_length=count*2, buffer=self.null_experience_list(count))
            self.assertEqual(buffer.experience_count, count)

    def test_dequeue(self):
        counts = [1, 5, 10, 20, 1000, 10000, 9999]
        for count in counts:
            buffer = ReplayBuffer(start_length=count//2, max_length=count*2, buffer=[idx for idx in range(count)])
            self.assertEqual(buffer.buffer[0], 0)
            self.assertEqual(buffer.buffer[count-1], count-1)
            for pop_count in range(count-1):
                buffer.dequeue()
                self.assertEqual(buffer.buffer[0], pop_count+1, "Buffer first value wrong - " +
                                 f"first value: {buffer.buffer[0]}, expected: {pop_count+1}")
                self.assertEqual(buffer.buffer[-1], count-1, "Buffer last value wrong - " +
                                 f"last value: {buffer.buffer[-1]}, expected: {count-1}")

    # def test_prep(self):

    #def test_is_full(self):

    #def test_is_ready(self):

    def test_training_items(self):
        # TODO test values are correct...
        self.fail()

    def test_append(self):
        count = 100
        start_length = count // 2
        max_length = count
        buffer = ReplayBuffer(start_length=start_length, max_length=max_length)
        for append_count in range(max_length*2):
            buffer.append(append_count)
            self.assertEqual(len(buffer.buffer), min(append_count+1, max_length), "Incorrect buffer size.")
            self.assertEqual(buffer.buffer[0], max(0, (append_count+1) - max_length), "Incorrect first value.")
            self.assertEqual(buffer.buffer[-1], append_count, "Incorrect last value.")

    #def test_sample(self):

    def test_random_indices(self):
        # TODO seed random to allow repeatable testing. Not too important right now
        start_length = max_length = 100
        buffer = ReplayBuffer(start_length=start_length,
                              max_length=max_length,
                              buffer=self.null_experience_list(start_length))
        for sample_size in range(0, start_length, 5):
            sample_indicies = buffer.random_indices(sample_size)
            self.assertEqual(len(sample_indicies), sample_size, "Incorrect size")
            for idx in sample_indicies:
                self.assertIn(idx, range(len(buffer.buffer)), "Invalid index")
            self.assertEqual(len(sample_indicies), len(set(sample_indicies)), "Not all elements unique")

    #def test_random_sample(self):

    #def test_update(self):  # TODO not implemented for this class type

    # def test_log(self): # TODO remove old code

    #def test_reservoir_sampling(self): # TODO remove old code

    #def test_multidimensional_shifting(self): # TODO test speed of this method

import midi
import os
import numpy as np
import pickle
import mingus.core.notes
import mingus.core.chords
import collections
import random
import copy
import cPickle
import yy_mh_util as mh_util
import yy_midi_util as midi_util


"""
There are several parameters in this file that only works for Nottingham;
e.g.
    Min, Max, Range;
    TrackNum = 3;
    MelodyTracki = 1;
    harmonyTracki = 2;
    ...
"""
VALID_TRACK_NUM = 3
MELODY_TRACK_I = 1
HARMONY_TRACK_I = 2
MELODY_MIN = 55
MELODY_MAX = 88
#MELODY_RANGE = MELODY_MAX - MELODY_MIN + 1


def get_good_files(input_dir, verbose=False):
    files = [ os.path.join(input_dir, f) for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f)) ] 
    good_files = []
    count_skip = 0
    for f in files:
        pattern = midi.read_midifile(f)
        if len(pattern) != VALID_TRACK_NUM:
            count_skip += 1
            continue
        good_files.append(f)
    if verbose:
        print "get_good_files() from {}, skip {} due to invalid track num, get {}".format(input_dir, count_skip, len(files)-count_skip)
    return good_files


def prepare_pickle(input_dir, output_fn, chord_cutoff=64, verbose=False, ref_chord_to_idx=None):
    """
    chord_cutoff: if chords are seen less than this cutoff, they are ignored and marked as
                  as rests in the resulting dataset
    ref_chord_to_idx: if this is None, it means prepare_pickle from the ground (e.g. prepare the train pickle);
        other wise, it means based on a certain map(e.g. one from train_pickle), prepare another pickle (e.g. test/valid);
    """
    store = {}

    data = []
    all_melody_twohot = []
    all_harmony_letter = []
    all_harmony_continuity = []
    song_names = []
    count_skip = 0
    for f in get_good_files(input_dir, verbose=verbose):
        if verbose:
            print "Start reading in {} ...".format(f)
        melody_twohot, multiple_note = mh_util.get_melody_twohot(f, MELODY_TRACK_I, MELODY_MIN, MELODY_MAX)
        if multiple_note:
            count_skip += 1
            continue
        harmony_letter, continuity = mh_util.get_harmony_letter(f, HARMONY_TRACK_I)
        all_melody_twohot.append(melody_twohot)
        all_harmony_letter.append(harmony_letter)
        all_harmony_continuity.append(continuity)
        song_names.append(f)
    if verbose:
        print "Skip {} due to multiple_note at a certain moment.".format(count_skip)

    if ref_chord_to_idx == None:
        chord_to_idx, idx_to_chord = mh_util.get_harmony_mapping(all_harmony_letter, chord_cutoff=chord_cutoff)
    else:
        chord_to_idx = ref_chord_to_idx
        idx_to_chord = {v: k for k, v in chord_to_idx.iteritems()}
    
    for i in range(len(all_melody_twohot)):
        harmony_twohot = mh_util.harmony_letter_to_twohot(all_harmony_letter[i], all_harmony_continuity[i], chord_to_idx)
        dt = mh_util.combine_mh(all_melody_twohot[i], harmony_twohot, chord_to_idx)
        data.append(dt)

    store["dim"] = data[0].shape[1]
    store["melody_min"] = MELODY_MIN
    store["melody_max"] = MELODY_MAX
    store["melody_dim"] = all_melody_twohot[0].shape[1]
    store["harmony_dim"] = store["dim"] - store["melody_dim"]
    store["chord_to_idx"] = chord_to_idx
    store["idx_to_chord"] = idx_to_chord
    store["data"] = data
    store["song_names"] = song_names

    if verbose:
        print "dim:", store["dim"]
        print "melody_min:", store["melody_min"]
        print "melody_max:", store["melody_max"]
        print "melody_dim:", store["melody_dim"]
        print "harmony_dim:", store["harmony_dim"]
        print "len(chord_to_idx):", len(store["chord_to_idx"])
        print "len(idx_to_chord):", len(store["idx_to_chord"])
        print "len(data):", len(store["data"])
        print "len(song_names):", len(store["song_names"])

    with open(output_fn, 'w') as f:
        cPickle.dump(store, f, protocol=-1)





class DataReader:
    def __init__(self, pickle_file):
        with open(pickle_file, 'r') as f:
            store = cPickle.load(f)
            self.dim = store["dim"]
            self.melody_min = store["melody_min"]
            self.melody_max = store["melody_max"]
            self.melody_dim = store["melody_dim"]
            self.harmony_dim = store["harmony_dim"]
            self.chord_to_idx = store["chord_to_idx"]
            self.idx_to_chord = store["idx_to_chord"]
            self.data = store["data"]
            self.song_names = store["song_names"]

    def get_batch_random(self, time_steps, batch_size):
        """
        return: matrix in [batch_size, time_steps, dim]
        """
        batch_data = np.zeros((batch_size, time_steps, self.dim))
        i = 0
        while i < batch_size:
            song_idx = random.randint(0, len(self.data)-1)
            song = self.data[song_idx]
            if len(song) < time_steps:
                continue
            start = random.randint(0, len(song) - time_steps)
            batch_data[i,:,:] = song[start:start+time_steps, :]
            i += 1
        return batch_data

    def get_batch_random_tbd(self, time_steps, batch_size):
        """
        return: matrix in [time_steps, batch_size, dim]
        """
        batch_data = self.get_batch_random_btd(time_steps, batch_size)
        return 

    def get_all_batches(self, time_steps, batch_size):
        """
        go through the song one by one, try to get as much as non-repeated batches as possible;
        return list[[batch_size, time_steps, dim]]
        """
        all_batches = []
        batch_data = np.zeros((batch_size, time_steps, self.dim))
        i = 0
        for song in self.data:
            for start_count in range(len(song)/time_steps):
                start = start_count * time_steps
                batch_data[i,:,:] = song[start:start+time_steps, :]
                i += 1
                if i == batch_size:
                    all_batches.append(copy.deepcopy(batch_data))
                    i = 0
        return all_batches





class DataWriter:
    def __init__(self, pickle_file):
        """
        pickle_file for metadata
        """
        with open(pickle_file, 'r') as f:
            store = cPickle.load(f)
            self.dim = store["dim"]
            self.melody_min = store["melody_min"]
            self.melody_max = store["melody_max"]
            self.melody_dim = store["melody_dim"]
            self.harmony_dim = store["harmony_dim"]
            self.chord_to_idx = store["chord_to_idx"]
            self.idx_to_chord = store["idx_to_chord"]

    def pickle_format_to_midi(self, song_seq, output_fn):
        """
        song_seq: 2d matrix: [T, dim]
        """
        self.twohot_mh_to_mid(song_seq[:,:self.melody_dim], song_seq[:,self.melody_dim:], output_fn)

    def twohot_mh_to_mid(self, melody_twohot, harmony_twohot, output_fn):
        melody_fn = output_fn + "_melody.mid"
        harmony_fn = output_fn + "_harmony.mid"

        melody_seq = mh_util.melody_twohot_to_seq(melody_twohot)
        harmony_seq = mh_util.harmony_twohot_to_seq(harmony_twohot, self.idx_to_chord)

        midi_util.noteStateMatrixToMidi(melody_seq, melody_fn, self.melody_min, self.melody_max)
        midi_util.noteStateMatrixToMidi(harmony_seq, harmony_fn, 0, 127)

        # combine these two midi files, with indicating instruments
        file_list = [melody_fn, harmony_fn]
        tracki_list = [0, 0]
        piano = 0
        bass = 33
        instrument_list = [piano, bass]
        midi_util.combine_midi(file_list, tracki_list, instrument_list, output_fn)




if __name__ == '__main__':
    # [1] prepare train_pickle
    input_dir = "data/Nottingham/train"
    output_fn = "data/Nottingham_train"
    prepare_pickle(input_dir, output_fn, verbose=True)

    # [2] prepare test/valid pickle
    # readin the train pickle as a reference
    train_pickle = "data/Nottingham_train"
    train_data = DataReader(train_pickle)
    chord_to_idx = train_data.chord_to_idx

    input_dir = "data/Nottingham/valid"
    output_fn = "data/Nottingham_valid"
    prepare_pickle(input_dir, output_fn, verbose=True, ref_chord_to_idx=chord_to_idx)
    
    input_dir = "data/Nottingham/test"
    output_fn = "data/Nottingham_test"
    prepare_pickle(input_dir, output_fn, verbose=True, ref_chord_to_idx=chord_to_idx)

    """
    # test convert back
    dr = DataReader("data/Nottingham_train")
    dw = DataWriter("data/Nottingham_train")
    dw.pickle_format_to_midi(dr.data[61], "outDWWWWWW")
    print dr.song_names[61]
    
    # test ABA convert
    harmony_letter, continuity = mh_util.get_harmony_letter("htest.mid", 0)
    harmony_twohot = mh_util.harmony_letter_to_twohot(harmony_letter, continuity, dr.chord_to_idx)
    harmony_seq = mh_util.harmony_twohot_to_seq(harmony_twohot, dr.idx_to_chord)
    midi_util.noteStateMatrixToMidi(harmony_seq, "htest_ABA.mid", 0, 127)
    """

    """
    [train_pickle meta]
    dim: 79
    melody_min: 55
    melody_max: 88
    melody_dim: 37
    harmony_dim: 42
    len(chord_to_idx): 40
    len(idx_to_chord): 40
    len(data): 667
    len(song_names): 667
    """


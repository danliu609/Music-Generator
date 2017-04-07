import midi
import os
import numpy as np
import mingus.core.notes
import mingus.core.chords
import collections
import random
import yy_midi_util as midi_util


# when use mingus to convert octave to int, the int value can only be in [0,11]
# this CHORD_BASE is for indicating the actually position of the octave
# it is from knowledge of the dataset
CHORD_BASE = 48  
CHORD_BLACKLIST = ['major third', 'minor third', 'perfect fifth']
NO_CHORD = 'NONE'
SHARPS_TO_FLATS = {
    "A#": "Bb",
    "B#": "C",
    "C#": "Db",
    "D#": "Eb",
    "E#": "F",
    "F#": "Gb",
    "G#": "Ab",
}


def resolve_chord(chord):
    """
    Resolves rare chords to their closest common chord, to limit the total
    amount of chord classes.
    """
    return chord    # yy: keep them all

    if chord in CHORD_BLACKLIST:
        return None
    # take the first of dual chords
    if "|" in chord:
        chord = chord.split("|")[0]
    # remove 7ths, 11ths, 9s, 6th,
    if chord.endswith("11"):
        chord = chord[:-2]
    if chord.endswith("7") or chord.endswith("9") or chord.endswith("6"):
        chord = chord[:-1]
    # replace 'dim' with minor
    if chord.endswith("dim"):
        chord = chord[:-3] + "m"
    return chord


def get_harmony_letter(midifile, tracki, verbose=False):
    """
    readin harmony track from midi file; convert it to letter notation;
    return list of letter notation of harmony; len(T);
        continuity;
    """
    harmony_sequence = midi_util.midiToNoteStateMatrix(midifile, tracki, 0, 127)
    harmony = []
    continuity = []  # record more things; True means continuous from prev;
    for i in range(harmony_sequence.shape[0]):
        notes = np.where(harmony_sequence[i, :128] == 1)[0]
        if len(notes) > 0:
            notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                # try flat combinations
                notes_shift = [ SHARPS_TO_FLATS[n] if n in SHARPS_TO_FLATS else n for n in notes_shift]
                chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            if len(chord) == 0:
                if verbose:
                    print "Could not determine chord: {} ({}), defaulting to last steps chord".format(notes_shift, i)
                if len(harmony) > 0:
                    harmony.append(harmony[-1])
                else:
                    harmony.append(NO_CHORD)
            else:
                resolved = resolve_chord(chord[0])
                if resolved:
                    harmony.append(resolved)
                else:
                    harmony.append(NO_CHORD)
        else:
            harmony.append(NO_CHORD)

        if np.count_nonzero(harmony_sequence[i, 128:]) > 0:  # having 1 means starting new;
            continuity.append(False)
        else:
            continuity.append(True)
    return np.asarray(harmony), continuity


def get_harmony_mapping(all_harmonies, chord_cutoff=64):
    """
    all_harmonies: all harmonies of the songs; dim: [num_songs * T], element are chord strings;
    chord_cutoff: if chords are seen less than this cutoff, they are ignored and marked as
                  as rests in the resulting dataset;
    return: harmony mapping;
    """
    chords = {}
    for harmony in all_harmonies:
        for h in harmony:
            if h not in chords:
                chords[h] = 1
            else:
                chords[h] += 1
    chords = { c: i for c, i in chords.iteritems() if chords[c] >= chord_cutoff }
    if NO_CHORD not in chords.keys():
        chords[NO_CHORD] = 1    # make sure NO_CHORD is in the map
    chord_to_idx = { c: i for i, c in enumerate(chords.keys()) }
    idx_to_chord = { i: c for i, c in enumerate(chords.keys()) }
    return chord_to_idx, idx_to_chord



def harmony_letter_to_twohot(harmony_letter, continuity, chord_to_idx):
    """
    letter list to one_hot list; + continuity;
    use the same way as midi2seq util: add two more cell;
    """
    harmony_dim = len(chord_to_idx) + 2  # two more: for indicating if the chord is continuity;
    harmony_twohot = np.zeros((harmony_letter.shape[0], harmony_dim))  
    harmony_idxs = [ chord_to_idx[h] if h in chord_to_idx else chord_to_idx[NO_CHORD] for h in harmony_letter ]
    for t,i in enumerate(harmony_idxs):
        harmony_twohot[t, i] = 1
        if continuity[t]:
            harmony_twohot[t, -2] = 1   # set continous bit
        else:
            harmony_twohot[t, -1] = 1   # set non-continous bit
    return harmony_twohot




def get_melody_twohot(midifile, tracki, pitchMin, pitchMax, verbose=False):
    """
    readin melody track from midi file; convert it to one_hot notation;
    add two more: one is empty; the other is continuity;
    return  dim [T * (melody_range + 3)], if multiple notes at a certain moment
    """
    span = pitchMax - pitchMin + 1
    melody_dim = span + 3  # three more: one for all empty; two for indicating if the chord is continuity;
    melody_sequence = midi_util.midiToNoteStateMatrix(midifile, tracki, pitchMin, pitchMax)
    melody_twohot = np.zeros((melody_sequence.shape[0], melody_dim))  
    multiple_note = False

    for t in range(melody_sequence.shape[0]):
        part1 = melody_sequence[t, :span]
        part2 = melody_sequence[t, span:]
        nonzero1 = np.count_nonzero(part1)
        nonzero2 = np.count_nonzero(part2)

        if nonzero1 == 0:
            melody_twohot[t, -3] = 1    # set empty bit
            melody_twohot[t, -1] = 1    # set non-continous bit
            continue
        if nonzero1 > 1:
            if verbose:
                print "Double note found: {}".format(input_filename)
            multiple_note = True
        melody_twohot[t, :-3] = part1

        if nonzero1 > 0 and nonzero2 == 0:
            melody_twohot[t, -2] = 1    # set continous bit
        else:
            melody_twohot[t, -1] = 1    # set non-continous bit
    return melody_twohot, multiple_note



def combine_mh(melody_twohot, harmony_twohot, chord_to_idx):
    """
    melody_twohot.shape may be different from harmony_twohot.shape; diff time length;
    """
    melody_time_len = melody_twohot.shape[0]
    harmony_time_len = harmony_twohot.shape[0]

    if melody_time_len == harmony_time_len:
        return np.column_stack((melody_twohot, harmony_twohot))

    time_len = max(melody_time_len, harmony_time_len)
    melody_dim = melody_twohot.shape[1]
    harmony_dim = harmony_twohot.shape[1]
    dim = melody_dim + harmony_dim
    combined = np.zeros((time_len, dim))

    combined[:melody_time_len, :melody_dim] = melody_twohot
    combined[:harmony_time_len, melody_dim:] = harmony_twohot

    if melody_time_len > harmony_time_len: 
        combined[harmony_time_len:, melody_dim + chord_to_idx[NO_CHORD]] = 1
        combined[harmony_time_len:, -1] = 1     # set harmony non-continous bit
    else:
        combined[melody_time_len:, melody_dim - 3] = 1    # no melody
        combined[melody_time_len:, melody_dim - 1] = 1    # set melody non-continous bit

    #print melody_twohot.shape
    #print harmony_twohot.shape
    #print combined.shape
    #raw_input()
    return combined




def melody_twohot_to_seq(melody_twohot):
    span = melody_twohot.shape[1] - 3  # three more: one for all empty; two for indicating if the chord is continuity;
    seq = np.zeros((melody_twohot.shape[0], span * 2))

    for t in range(melody_twohot.shape[0]):
        seq[t,:span] = melody_twohot[t,:-3]
        continuity = melody_twohot[t,-2]
        if continuity == 0:
            seq[t,span:] = seq[t,:span]
    return seq




def harmony_twohot_to_seq(harmony_twohot, idx_to_chord):
    assert harmony_twohot.shape[1] == len(idx_to_chord) + 2
    span = 128
    seq = np.zeros((harmony_twohot.shape[0], span * 2))
    for t in range(harmony_twohot.shape[0]):
        idx = np.argmax(harmony_twohot[t,:-2])
        if idx not in idx_to_chord:
            raise Exception("No chord index found: {}".format(idx))
        shorthand = idx_to_chord[idx]
        if shorthand == NO_CHORD:
            continue

        chord = mingus.core.chords.from_shorthand(shorthand)
        # the chord number can only be in range [0,11], but it should be in the increasing order
        # by looking back at the prev number, we can increase the base if needed.
        prev_i = 0
        base = CHORD_BASE
        for note in chord:
            i = int(mingus.core.notes.note_to_int(note)) + base
            if i < prev_i:
                i += 12
                base += 12
            seq[t,i] = 1
            prev_i = i
        
        continuity = harmony_twohot[t,-2]
        if continuity == 0:
            seq[t,span:] = seq[t,:span]
    return seq
    




if __name__ == '__main__':
    #notes = [31, 35, 38]
    notes = [26, 30, 33, 36]
    notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
    chord = mingus.core.chords.determine(notes_shift, shorthand=True)
    if len(chord) == 0:
        # try flat combinations
        notes_shift = [ SHARPS_TO_FLATS[n] if n in SHARPS_TO_FLATS else n for n in notes_shift]
        chord = mingus.core.chords.determine(notes_shift, shorthand=True)
    if len(chord) == 0:
        print "Could not determine chord: {}, defaulting to last steps chord".format(notes_shift)
    else:
        print chord
        resolved = resolve_chord(chord[0])
        print "[R]", resolved
        #recover
        rec = mingus.core.chords.from_shorthand(resolved)
        for n in rec:
            print mingus.core.notes.note_to_int(n)
            #CHORD_BASE




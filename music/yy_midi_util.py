import midi
import numpy as np
import copy

VELOCITY = 80

#matrix to midi
def noteStateMatrixToMidi(statematrix, output_fn, pitchMin, pitchMax):
    lowerBound = pitchMin
    upperBound = pitchMax + 1
    span = upperBound-lowerBound    # The note range
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upperBound-lowerBound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=VELOCITY, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    if not output_fn.endswith(".mid"):
        output_fn += ".mid"
    midi.write_midifile("{}".format(output_fn), pattern)



#midi to matrix
#time * 176(88*2)
#2 represents states 
#(0,0) off, (1,1) on, (1,0) continue  
def midiToNoteStateMatrix(midifile, tracki, pitchMin, pitchMax, squash=True):
    lowerBound = pitchMin
    upperBound = pitchMax + 1
    span = upperBound-lowerBound    # The note range
    try:
        pattern = midi.read_midifile(midifile)
    except:
        pass
    #timeleft = [track[0].tick for track in pattern]
    #posns = [0 for track in pattern]
    timeleft = pattern[tracki][0].tick
    posns = 0
    statematrix = []
    time = 0

    state = [[0,0] for x in range(span)]
    statematrix.append(state)
    condition = True
    #print len([track[0].tick for track in pattern])
    #print pattern.resolution
    #print timeleft
   
    while condition:
        if time % (pattern.resolution / 4) == (pattern.resolution / 8):
           # print time
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0],0] for x in range(span)]
            statematrix.append(state)
         
        #for i in range(len(timeleft)): #For each track
        if not condition:
            break
        while timeleft == 0:
            track = pattern[tracki]
            pos = posns

            evt = track[pos]
            if isinstance(evt, midi.NoteEvent):
                if (evt.pitch < lowerBound) or (evt.pitch >= upperBound):
                    pass
                        # print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                else:
                        
                    if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                        state[evt.pitch-lowerBound] = [0, 0]
                            
                    else:
                        state[evt.pitch-lowerBound] = [1, 1]
            elif isinstance(evt, midi.TimeSignatureEvent):
                if evt.numerator not in (2, 4):
                    # We don't want to worry about non-4 time signatures. Bail early!
                    # print "Found time signature event {}. Bailing!".format(evt)
                    out =  statematrix
                    condition = False
                    break
            try:
                timeleft = track[pos + 1].tick
                posns += 1
            except IndexError:
                timeleft = None

        if timeleft is not None:
            timeleft -= 1

        if timeleft is None:
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    #statematrix = np.asarray(statematrix).tolist()
    statematrix = np.asarray(statematrix)
    return statematrix



def combine_midi(file_list, tracki_list, instrument_list, output_fn):
    """
    see https://zh.wikipedia.org/wiki/General_MIDI    
    index shift by -1;
    """
    if len(file_list) != len(tracki_list) or len(file_list) != len(instrument_list):
        raise Exception("Length doesn't match.")
    pattern = midi.Pattern()
    for i,f in enumerate(file_list):
        track = midi.read_midifile(f)[tracki_list[i]]
        track.insert(0, midi.ProgramChangeEvent(tick = 0, channel = i, data = [instrument_list[i]]))
        pattern.append(track)
    if not output_fn.endswith(".mid"):
        output_fn += ".mid"
    midi.write_midifile(output_fn, pattern)









def unpack_seq(seq):
    span = len(seq[0])/2
    seq = np.array(seq)
    seq = np.dstack((seq[:, :span], seq[:, span:]))
    return np.asarray(seq)

def empty_unpacked_seq(ref):
    return np.zeros((ref.shape[0], ref.shape[1], 2))

def pack_seq(seq):
    seq = np.array(seq)
    seq = np.hstack((seq[:, :, 0], seq[:, :, 1]))
    return np.asarray(seq)

def shift_note(midifile, tracki, pitchLowerBound, pitchUpperBound):
    ori_seq = midiToNoteStateMatrix(midifile, tracki, 0, 127)
    seq = unpack_seq(ori_seq)
    new_seq = empty_unpacked_seq(ref=seq)

    # get min/max pitch
    minPitch = 127
    maxPitch = 0
    for t,moment_notes in enumerate(seq):
        for i in range(len(moment_notes)):
            if moment_notes[i][0] == 1:
                minPitch = min(minPitch,i)
                maxPitch = max(maxPitch,i)

    if minPitch < pitchLowerBound or maxPitch > pitchUpperBound:
        print "midi_util.shift_note: shift minPitch to lowerBound {}.".format(pitchLowerBound)
        sh = pitchLowerBound - minPitch
        for t,moment_notes in enumerate(seq):
            for i in range(len(moment_notes)):
                if moment_notes[i][0] == 1 and i+sh < new_seq.shape[1]:
                    new_seq[t][i+sh][0] = moment_notes[i][0]
                    new_seq[t][i+sh][1] = moment_notes[i][1]
        new_seq = pack_seq(new_seq)
    else:
        print "midi_util.shift_note: no shift needed."
        new_seq = ori_seq

    output_fn = midifile + "_shifted.mid"
    noteStateMatrixToMidi(new_seq, output_fn, 0, 127)
    return output_fn





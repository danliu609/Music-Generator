#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 17:05:43 2016

@author: ztan6
"""
import midi
import os
import numpy as np
import pickle
import mingus.core.notes
import mingus.core.chords
import collections
import random

lowerBound = 55 #The lowest note
upperBound = 89 #The highest note
span = upperBound-lowerBound #The note range
chord_num = 48

#matrix to midi
def noteStateMatrixToMidi(statematrix, name, span=span):
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
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lowerBound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(name), pattern)

#midi to matrix
#time * 176(88*2)
#2 represents states 
#(0,0) off, (1,1) on, (1,0) continue  
def midiToNoteStateMatrix(midifile, tracki, squash=True, span=span):
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
    print len([track[0].tick for track in pattern])
    print pattern.resolution
    print timeleft
   
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
    statematrix = np.asarray(statematrix).tolist()
    return statematrix

#get the full path of the dir
#called in def getRightData(path)
def listdir_fullpath(d):
    paths = []
    for f in os.listdir(d):
        if f.startswith('.'):
            continue # ignore hidden files that starts with '.'
        paths.append(os.path.join(d, f))
    return paths

#remove the bad midi header file
def getRightData(path):
    paths = []
    listPath = listdir_fullpath(path)
    for path in listPath:
        if path.startswith('.'):
            continue # ignore hidden files that starts with '.'
        try:
            pattern = midi.read_midifile(path)
        except:
            pass
        if(len(pattern) == 3):
            print path
            paths.append(path)
    return paths

#get the melody and harmony of the songs in the paths
#as input and output for the model
def getDataset(listPath):
    inputs = []
    outputs = []
    for path in listPath:
        if path.startswith('.'):
            continue # ignore hidden files that starts with '.'
        
        print path
        statematrix1 = midiToNoteStateMatrix(path, 1)
        statematrix2 = midiToNoteStateMatrix(path, 2)
        inputs.append(statematrix1)
        outputs.append(statematrix2)
    return inputs,outputs

#mapping the chord to index
#make a dictionay 
def count_dictionary(outputs):
    chords = []
    for statematrix in outputs:
        for harmony_sequence in statematrix:
            notes = []
            #harmony_sequence = [0, 0, 0, 0, 0,  0 , 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            #harmony_sequence = j
            #notes = harmony_sequence.index(1)
            for i in range(len(harmony_sequence)):
                if harmony_sequence[i] == 1:
                    notes.append(i)
                    #notes  = np.where(harmony_sequence == 1)
            print notes
            #if len(notes) > 0:
            notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            print notes_shift
            print tuple(sorted(notes))
            chords.append(tuple(sorted(notes)))
            #chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            #print chord
    count = [['*UNK*', -1]]
    count.extend(collections.Counter(chords).most_common(chord_num - 1))
    dictionary = dict()
    for chord, _ in count:
        dictionary[chord] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary


'''
def mapping(outputs):    
    dictionary, reverse_dictionary = count_dictionary(outputs)
    for statematrix2 in outputs:
        for j in statematrix2:
            notes = []
            harmony_sequence = j
            #notes = harmony_sequence.index(1)
            for i in range(len(harmony_sequence)):
                if harmony_sequence[i] == 1:
                    notes.append(i)
                    #notes  = np.where(harmony_sequence == 1)
            print notes
            #if len(notes) > 0:
            notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            print notes_shift
            print tuple(sorted(notes))
            chord = mingus.core.chords.determine(notes_shift, shorthand=True)
            print chord
            if tuple(sorted(notes)) not in dictionary:
                dictionary[tuple(sorted(notes))] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary
'''


def harmony(dictionary, outputs):
    harmony = []
    for statematrix in outputs:
        song = []
        for j in statematrix:
                notes = []
            
                harmony_sequence = j
                #notes = harmony_sequence.index(1)
                for i in range(len(harmony_sequence)):
                    if harmony_sequence[i] == 1:
                        notes.append(i)
                        #notes  = np.where(harmony_sequence == 1)
                print notes
                #if len(notes) > 0:
                notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
                #print notes_shift
                if tuple(sorted(notes)) in dictionary:
                    print dictionary[tuple(sorted(notes))]
                    song.append(dictionary[tuple(sorted(notes))])
                else:
                    print dictionary['*UNK*']
                    song.append(dictionary['*UNK*'])
        harmony.append(song)
    return harmony

def harmony_index(dictionary, har):
    song = []
    for j in har:
        notes = []
        harmony_sequence = j
            #notes = harmony_sequence.index(1)
        for i in range(len(harmony_sequence)):
            if harmony_sequence[i] == 1:
                notes.append(i)
                        #notes  = np.where(harmony_sequence == 1)
        print notes
            #if len(notes) > 0:
            #notes_shift = [ mingus.core.notes.int_to_note(h%12) for h in notes]
            #print notes_shift
        if tuple(sorted(notes)) in dictionary:
            print dictionary[tuple(sorted(notes))]
            song.append(dictionary[tuple(sorted(notes))])
        else:
            print dictionary['*UNK*']
            song.append(dictionary['*UNK*'])
    return song

def melody_indexs(melody_train):
    melody_index = []
    for melody in melody_train:
        song = []
        for timei in melody:
            note_index = 0
            for i in range(span):
                if timei[i] == 1:
                    note_index = i+1
            song.append(note_index)
        melody_index.append(song)
    return melody_index
        
    
def getbatch(batchsz, num_steps, dataset_input, dataset_output):
    batch1 = []
    batch2 = []
    for i in range(batchsz):
        index = random.randint(0, len(dataset_input)-1)
        print index
        matrix1 = dataset_input[index]
        matrix2 = dataset_output[index]
        start = random.randint(0, len(matrix1)- num_steps + 1 - 1)
        inpt = []
        for j in range(num_steps):
            inpt.append(matrix1[start+j])
        oupt = matrix2[start+1: start+1+num_steps]
        batch1.append(inpt)
        batch2.append(oupt)
    inputs = np.reshape(batch1,[batchsz, num_steps, len(dataset_input[0][0]) ])
    outputs = np.reshape(batch2, [batchsz, num_steps])
    return inputs, outputs

def get_test_self(name, track):
    melody_test = midiToNoteStateMatrix("{}.mid".format(name) ,track)
    f = open('{}.pckl'.format(name), 'wb')
    pickle.dump(melody_test, f)
    f.close()

def combine_midi(midi1, midi2, instrument1, instrument2, name):
    pattern1 = midi.read_midifile(midi1)
    pattern2 = midi.read_midifile(midi2)
    pattern1.append(pattern2[0])
    pattern1[0].insert(0, midi.ProgramChangeEvent(tick = 0, channel = 0, data = [instrument1]))
    pattern1[1].insert(0, midi.ProgramChangeEvent(tick = 0, channel = 0, data = [instrument2]))
    midi.write_midifile("{}.mid".format(name), pattern1)
    

#get the right path of the nottingham dataset
paths  = getRightData("Nottingham/train")
paths1 = getRightData("Nottingham/test")
paths2 = getRightData("Nottingham/valid")
paths_all = paths+paths1+paths2
#save the paths_all
f = open('paths_all.pckl', 'wb')
pickle.dump(paths_all, f)
f.close()
#get the 700 as the training set
paths_train = paths[0:700]
#save the training_set pahths
f = open('paths_train.pckl', 'wb')
pickle.dump(paths_train, f)
f.close()

#get the melody and harmony of training set
#and the data is on when data is 1
inputs_train, outputs_train = getDataset(paths_train)
f = open('melody_train.pckl', 'wb')
pickle.dump(inputs_train, f)
f.close() 
f = open('outputs_train.pckl', 'wb')
pickle.dump(outputs_train, f)
f.close()

#get the melody index of trainint set
melody_index = melody_indexs(inputs_train)
f = open('melody_index.pckl', 'wb')
pickle.dump(melody_index, f)

#get the harmony dictionary and save it
dictionary, reverse_dictionary = count_dictionary(outputs_train)
f = open('dictionary.pckl', 'wb')
pickle.dump(dictionary, f)
f.close()
f = open('reverse_dictionary.pckl', 'wb')
pickle.dump(reverse_dictionary, f)
f.close()

#get the index of the harmony of training set
harmony_train = harmony(dictionary, outputs_train)
f = open('harmony_train.pckl', 'wb')
pickle.dump(harmony_train, f)
f.close()

#get one hot vector of harmony
harmony_one_hot = []
for matrix in harmony_train:
    song_onehot = []    
    for index in matrix:
        one_hot = [0]*48
        one_hot[index] = 1
        song_onehot.append(one_hot)
    harmony_one_hot.append(song_onehot)

f = open('harmony_onehot.pckl', 'wb')
pickle.dump(harmony_one_hot, f)
f.close()



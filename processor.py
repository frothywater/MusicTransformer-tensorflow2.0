import mido

beat_division = 16
unit_per_bar = beat_division * 4
init_tempo = 500000
default_velocity = 64


class Event:
    def __init__(self, type: str, value=None) -> None:
        self.type = type
        self.value = value

    def __str__(self) -> str:
        return f"{self.type}_{self.value}"


def create_dict():
    events = []
    note_names = ['C', 'Db', 'D', 'Eb', 'E',
                  'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    chord_qualities = ['maj', 'min', 'dim', 'aug',
                       'maj7', 'min7', '7', 'm7b5', 'dim7', 'sus']

    events += [Event("note_on", i) for i in range(128)]
    events += [Event("note_off", i) for i in range(128)]
    events += [Event("chord", f"{root}_{quality}")
               for root in note_names for quality in chord_qualities]
    events += [Event("time_shift", i+1) for i in range(unit_per_bar)]
    events += [Event("velocity", i) for i in range(128)]
    events += [Event("pad"), Event("sos"), Event("eos")]

    str2word = {str(event): i for i, event in enumerate(events)}
    word2event = {i: event for i, event in enumerate(events)}
    return str2word, word2event


str2word, word2event = create_dict()
pad_word = str2word["pad"]
sos_word = str2word["sos"]
eos_word = str2word["eos"]
dictionary_size = len(str2word)


def encode_midi(path: str):
    midi = mido.MidiFile(path)
    events = []
    for msg in mido.merge_tracks(midi.tracks):
        if msg.time > 0:
            delta = round(msg.time / midi.ticks_per_beat * beat_division)
            if delta > 0 and delta <= unit_per_bar:
                events.append(Event("time_shift", delta))
            elif delta > unit_per_bar:
                while delta > 0:
                    this_delta = unit_per_bar if delta > unit_per_bar else delta
                    events.append(Event("time_shift", this_delta))
                    delta -= this_delta
        if msg.type == "note_on" and msg.velocity > 0:
            events.append(Event("velocity", msg.velocity))
            events.append(Event("note_on", msg.note))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            events.append(Event("note_off", msg.note))
        elif msg.is_meta and msg.type == "marker" and len(msg.text.split("_")) == 2:
            events.append(Event("chord", msg.text))
    words = [str2word[str(event)] for event in events]
    return words


def decode_midi(words: list, path: str):
    messages = []
    next_time_shift = 0
    current_velocity = default_velocity
    for word in words:
        event = word2event[word]
        if event.type == "time_shift":
            next_time_shift += event.value
        elif event.type == "velocity":
            current_velocity = event.value
        else:
            if event.type == "note_on":
                messages.append(mido.Message(
                    "note_on", note=event.value, velocity=current_velocity, time=next_time_shift))
            elif event.type == "note_off":
                messages.append(mido.Message(
                    "note_off", note=event.value, time=next_time_shift))
            elif event.type == "chord":
                messages.append(mido.MetaMessage(
                    "marker", text=event.value, time=next_time_shift))
            else:
                continue
            next_time_shift = 0

    midi = mido.MidiFile()
    midi.ticks_per_beat = beat_division
    track = mido.MidiTrack()
    track.extend(messages)
    midi.tracks.append(track)
    midi.save(path)

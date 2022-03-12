import math
from itertools import accumulate

import mido

# Exported
beat_division = 4
unit_per_bar = beat_division * 4
init_tempo = 500000

pitch_range = range(48,84)  # C3-C5
note_names = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
chord_qualities = ['M','m','o','+','MM7','Mm7','mM7','mm7','o7','%7','+7','M7','sus']

velocity_start = 2
velocity_step = 4
velocity_range = range(velocity_start, 128 + 1, velocity_step)
default_velocity = 64


class Event:
    def __init__(self, type: str, value=None) -> None:
        self.type = type
        self.value = value

    def __str__(self) -> str:
        return f"{self.type}_{self.value}"


def create_dict():
    events = []

    events += [Event("note_on", i) for i in pitch_range]
    events += [Event("note_off", i) for i in pitch_range]
    events += [Event("chord", f"{root}_{quality}") for root in note_names for quality in chord_qualities]
    events += [Event("time_shift", i + 1) for i in range(unit_per_bar)]
    events += [Event("velocity", i) for i in velocity_range]
    events += [Event("pad"), Event("sos"), Event("eos")]

    str2word = {str(event): i for i, event in enumerate(events)}
    word2event = {i: event for i, event in enumerate(events)}
    return str2word, word2event


# Exported
str2word, word2event = create_dict()
pad_word = str2word["pad_None"]
sos_word = str2word["sos_None"]
eos_word = str2word["eos_None"]
dictionary_size = len(str2word)


def round_velocity(v: int) -> int:
    return math.trunc((v - velocity_start) / velocity_step) * velocity_step + velocity_start


def sorted_messages(msgs: list) -> list:
    time_list = [msg.time for msg in msgs]
    absolute_time_list = accumulate(time_list)
    is_meta_list = [msg.is_meta for msg in msgs]
    tuples = zip(msgs, absolute_time_list, is_meta_list)
    # Sort first by absolute time asc., then by is_meta dsc.
    sorted_tuples = sorted(tuples, key=lambda t: (t[1], -t[2]))
    return [tuple[0] for tuple in sorted_tuples]


def encode_midi(path: str) -> list:
    midi = mido.MidiFile(path)
    events = []
    msgs = sorted_messages(mido.merge_tracks(midi.tracks))
    for msg in msgs:
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
            rounded_velocity = round_velocity(msg.velocity)
            events.append(Event("velocity", rounded_velocity))
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
                messages.append(mido.Message("note_on", note=event.value, velocity=current_velocity, time=next_time_shift))
            elif event.type == "note_off":
                messages.append(mido.Message("note_off", note=event.value, time=next_time_shift))
            elif event.type == "chord":
                messages.append(mido.MetaMessage("marker", text=event.value, time=next_time_shift))
            else:
                continue
            next_time_shift = 0

    midi = mido.MidiFile(ticks_per_beat=beat_division)
    midi.tracks.append(mido.MidiTrack())  # Add a dummy track
    track = mido.MidiTrack(messages)
    midi.tracks.append(track)
    midi.save(path)


# Utils
def write_messages_to_text(msgs: list, path: str):
    with open(path, "w") as file:
        for msg in msgs:
            file.write(f"{msg}\n")


def write_midi_to_text(path_midi: str, path_text: str):
    midi = mido.MidiFile(path_midi)
    msgs = mido.merge_tracks(midi.tracks)
    write_messages_to_text(msgs, path_text)


def write_dict(path: str):
    with open(path, "w") as file:
        file.write(str(str2word))


def test():
    write_dict("data/dict.txt")

    file = "1"
    test_file = f"data/{file}.mid"
    write_midi_to_text(test_file, f"data/{file}_original.txt")

    midi = mido.MidiFile(test_file)
    msgs = sorted_messages(mido.merge_tracks(midi.tracks))
    write_messages_to_text(msgs, f"data/{file}_sorted.txt")

    around_file = f"data/{file}_around.mid"
    decode_midi(encode_midi(test_file), around_file)
    write_midi_to_text(around_file, f"data/{file}_around.txt")


if __name__ == "__main__":
    test()

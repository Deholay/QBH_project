import numpy as np

def midi_to_simplified_notation(midi_numbers):
    
    # Maps for each octave range
    notation_map_default = {
        48: '1', 50: '2', 52: '3', 53: '4',
        55: '5', 57: '6', 59: '7'
    }
    notation_map_lower = {
        36: 'A', 38: 'B', 40: 'C', 41: 'D',
        43: 'E', 45: 'F', 47: 'G'
    }
    notation_map_upper = {
        60: 'H', 62: 'I', 64: 'J', 65: 'K',
        67: 'L', 69: 'M', 71: 'N'
    }

    simplified_notation = []

    # Round MIDI numbers to the nearest integer
    midi_numbers = [round(midi) for midi in midi_numbers]
    
    for midi in midi_numbers:
        # Check the range and map to the corresponding notation
        if 48 <= midi <= 59:  # Default octave
            simplified_notation.append(notation_map_default.get(midi, '?'))
        elif 36 <= midi <= 47:  # Lower octave
            simplified_notation.append(notation_map_lower.get(midi, '?'))
        elif 60 <= midi <= 71:  # Higher octave
            simplified_notation.append(notation_map_upper.get(midi, '?'))
        else:
            simplified_notation.append('?')  # Out of range

    return simplified_notation


def print_song_info(song_name, midi_numbers, beats):

    # Convert MIDI numbers to simplified notation
    simplified_notation = midi_to_simplified_notation(midi_numbers)
    notation_str = ''.join([str(notation) for notation in simplified_notation])

    # Convert beats to string
    beats = [int(2 ** np.log2(beat)) for beat in beats]
    beats_str = ''.join([str(beat) for beat in beats])

    # Print the formatted song information
    print(f"{song_name}\t{notation_str}/{beats_str}")

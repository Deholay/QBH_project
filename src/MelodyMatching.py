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

def simplified_notation_to_midi(simplified_notation):
    """
    Convert simplified notation to MIDI numbers and calculate MIDI differences.
    """
    # Mapping for simplified notation to MIDI
    notation_to_midi = {
        '1': 48, '2': 50, '3': 52, '4': 53,
        '5': 55, '6': 57, '7': 59,
        'A': 36, 'B': 38, 'C': 40, 'D': 41,
        'E': 43, 'F': 45, 'G': 47,
        'H': 60, 'I': 62, 'J': 64, 'K': 65,
        'L': 67, 'M': 69, 'N': 71
    }

    # Convert simplified notation to MIDI numbers
    midi_numbers = []
    for note in simplified_notation:
        midi = notation_to_midi.get(note)
        if midi is not None:
            midi_numbers.append(midi)
        else:
            raise ValueError(f"Invalid note in simplified notation: {note}")

    # Calculate MIDI differences
    midi_differences = [
        midi_numbers[i] - midi_numbers[i - 1]
        for i in range(1, len(midi_numbers))
    ]

    return midi_numbers, midi_differences

def print_song_info(song_name, midi_numbers, beats):

    # Convert MIDI numbers to simplified notation
    simplified_notation = midi_to_simplified_notation(midi_numbers)
    notation_str = ''.join([str(notation) for notation in simplified_notation])

    # Convert beats to string
    beats = [int(2 ** np.log2(beat)) for beat in beats]
    beats_str = ''.join([str(beat) for beat in beats])

    # Print the formatted song information
    print(f"{song_name}\t{notation_str}/{beats_str}")
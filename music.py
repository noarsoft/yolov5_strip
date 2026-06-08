import numpy as np
import simpleaudio as sa

NOTE_FREQS = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
}

def generate_sine_wave(freq, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    return wave

def play_wave(wave, sample_rate=44100):
    wave = (wave * 32767).astype(np.int16)  #PCM
    play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
    play_obj.wait_done()

def play_notes(notes, duration=0.5):
    for note in notes:
        freq = NOTE_FREQS[note]
        wave = generate_sine_wave(freq, duration)
        play_wave(wave)


notes_sequence = ["C4", "D4", "E4", "F4", "G4"]
play_notes(notes_sequence, duration=0.5)

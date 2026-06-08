import numpy as np
import simpleaudio as sa

NOTE_FREQS = {
    "C4": 261.63,  # โด
    "D4": 293.66,  # เร
    "E4": 329.63,  # มี
    "F#4": 369.99,  # ฟา#
    "G4": 392.00,  # ซอล
    "A4": 440.00,  # ลา
    "C#5": 554.37,  # โด# สูง
    "REST": 0  # เงียบ
}


def generate_karplus_strong_wave(freq, duration, sample_rate=44100, damping=0.996):
    """ ใช้ Karplus-Strong Algorithm สำหรับกีตาร์โปร่ง & คลาสสิก """
    if freq == 0:
        return np.zeros(int(sample_rate * duration))

    n_samples = int(sample_rate * duration)
    buffer_size = int(sample_rate / freq)

    buffer = np.random.uniform(-1, 1, buffer_size)
    wave = np.zeros(n_samples)

    for i in range(n_samples):
        wave[i] = buffer[0]
        avg = 0.5 * (buffer[0] + buffer[1])
        avg *= damping
        buffer[:-1] = buffer[1:]
        buffer[-1] = avg

    return wave


def generate_piano_wave(freq, duration, sample_rate=44100):
    """ ใช้ Wave Table Synthesis สำหรับเสียงเปียโน """
    if freq == 0:
        return np.zeros(int(sample_rate * duration))

    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # **สร้าง Wave Table (คล้ายซาวด์เปียโน)**
    wave = np.sin(2 * np.pi * freq * t)  # Base Sine Wave
    wave += 0.5 * np.sin(2 * np.pi * 2 * freq * t)  # Harmonic 2nd
    wave += 0.25 * np.sin(2 * np.pi * 3 * freq * t)  # Harmonic 3rd

    # **ใช้ ADSR Envelope เพื่อให้เสียงสมจริงขึ้น**
    attack = int(0.05 * len(wave))  # 5% ของโน้ตสำหรับ Attack
    decay = int(0.15 * len(wave))  # 15% สำหรับ Decay
    sustain_level = 0.7
    release = int(0.3 * len(wave))  # 30% สำหรับ Release

    envelope = np.ones(len(wave))
    envelope[:attack] = np.linspace(0, 1, attack)  # Attack
    envelope[attack:attack + decay] = np.linspace(1, sustain_level, decay)  # Decay
    envelope[-release:] = np.linspace(sustain_level, 0, release)  # Release

    wave *= envelope  # Apply Envelope
    return wave


def apply_classical_guitar_effects(wave, sample_rate=44100):
    """ ใช้ Low-pass Filter ปรับเสียงให้เป็น Classical Guitar """
    cutoff = 2000
    fft_wave = np.fft.rfft(wave)
    freqs = np.fft.rfftfreq(len(wave), d=1 / sample_rate)
    fft_wave[freqs > cutoff] = 0
    wave_classical = np.fft.irfft(fft_wave)

    wave_classical *= 1.2
    return wave_classical


def play_wave(wave, sample_rate=44100):
    wave = (wave * 32767).astype(np.int16)
    play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
    play_obj.wait_done()


def play_notes_with_durations(notes, durations, effect=''):
    if len(notes) != len(durations):
        raise ValueError("The number of notes and durations must be the same.")

    for note, duration in zip(notes, durations):
        freq = NOTE_FREQS.get(note, 440)

        if effect == 'acoustic':
            wave = generate_karplus_strong_wave(freq, duration, damping=0.996)  # Acoustic Guitar
        elif effect == 'classical':
            wave = generate_karplus_strong_wave(freq, duration, damping=0.999)  # Classical Guitar
            wave = apply_classical_guitar_effects(wave)
        elif effect == 'piano':
            wave = generate_piano_wave(freq, duration)  # ใช้ Piano Wave Table Synthesis
        else:
            wave = generate_karplus_strong_wave(freq, duration)

        play_wave(wave)


# https://www.bloggang.com/m/viewdiary.php?id=thaiger&month=09-2020&date=22&group=7&gblog=15
notes_sequence = [
    "G4", "REST", "G4", "REST", "G4",  # ซอล (G4) สองครั้ง
    "REST", "G4", "E4", "REST",  # ซอล - มี
    "D4", "E4", "REST", "G4", "C4",  # เร - มี - ซอล - โด

    "REST", "G4", "E4", "REST",  # ซอล - มี
    "D4", "E4", "REST", "C4", "D4",  # เร - มี - โด - เร
    "C4", "A4", "REST", "C4", "C4",  # โด - ลา - โด - โด
    "A4", "G4", "REST", "C4", "REST",  # ลา - ซอล - โด
]

durations = [
    0.5,  0.5, 0.5, 0.5, 0.5,  # ซอล (G4) สองครั้ง
    0.5, 0.25, 0.25, 0.5,  # ซอล - มี
    0.25, 0.25, 0.25, 0.5, 0.5,  # เร - มี - ซอล - โด

    0.5, 0.25, 0.25, 0.5,  # ซอล - มี
    0.25, 0.25, 0.25, 0.5, 0.5,  # เร - มี - โด - เร
    0.25, 0.25, 0.25, 0.5, 0.5,  # โด - ลา - โด - โด
    0.25, 0.25, 0.25, 0.5, 0.5  # ลา - ซอล - โด
]

# # เมโลดี้สำหรับ "Her Feline Friend"

#
# play_notes_with_durations(notes_sequence, durations)


play_notes_with_durations(notes_sequence, durations, effect='acoustic')
# play_notes_with_durations(notes_sequence, durations, effect='piano')



# import numpy as np
# import simpleaudio as sa
#
# NOTE_FREQS = {
#     "C4": 261.63,  # โด
#     "D4": 293.66,  # เร
#     "E4": 329.63,  # มี
#     "F#4": 369.99,  # ฟา# (ใช้สำหรับเมโลดี้ของคุณ)
#     "G4": 392.00,  # ซอล
#     "A4": 440.00,  # ลา
#     "C#5": 554.37,  # โด# (สูงกว่า C4)
#     "REST": 0  # เงียบ
# }
#
# def generate_sine_wave(freq, duration, sample_rate=44100):
#     if freq == 0:
#         return np.zeros(int(sample_rate * duration))
#     t = np.linspace(0, duration, int(sample_rate * duration), False)
#     wave = 0.5 * np.sin(2 * np.pi * freq * t)
#     return wave
#
# def play_wave(wave, sample_rate=44100):
#     wave = (wave * 32767).astype(np.int16)  # PCM
#     play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
#     play_obj.wait_done()
#
# def play_notes_with_durations(notes, durations):
#     if len(notes) != len(durations):
#         raise ValueError("The number of notes and durations must be the same.")
#
#     for note, duration in zip(notes, durations):
#         freq = NOTE_FREQS.get(note, 440)  # ค่า default คือ A4 (440 Hz)
#         wave = generate_sine_wave(freq, duration)
#         play_wave(wave)
#
# # เมโลดี้สำหรับ "Her Feline Friend"
# notes_sequence = [
#     "F#4", "A4",  # "Her feline friend"
#     "A4", "C#5",  # "by her side"
#     "A4", "C#5",  # "They dance"
#     "C#5", "A4",  # "through the night"
#
#     "E4", "G4",  # "Whiskers and"
#     "G4", "REST",  # "meows"
#     "D4", "F#4",  # "Such a"
#     "D4", "F#4",  # "precious sight"
#
#     "D4", "F#4",  # "Their love"
#     "F#4", "A4",  # "so pure"
#     "A4", "C#5",  # "It's a bond"
#     "A4", "C#5",  # "untamed"
#
#     "D4", "F#4",  # "In perfect"
#     "F#4", "A4",  # "harmony"
#     "A4", "C#5",  # "Their hearts"
#     "A4", "C#5"  # "are the same"
# ]
#
# durations = [
#     0.5, 0.5,  # "Her feline friend"
#     0.5, 0.5,  # "by her side"
#     0.5, 0.5,  # "They dance"
#     0.5, 0.5,  # "through the night"
#
#     0.5, 0.5,  # "Whiskers and"
#     0.5, 0.5,  # "meows"
#     0.5, 0.5,  # "Such a"
#     0.5, 0.5,  # "precious sight"
#
#     0.5, 0.5,  # "Their love"
#     0.5, 0.5,  # "so pure"
#     0.5, 0.5,  # "It's a bond"
#     0.5, 0.5,  # "untamed"
#
#     0.5, 0.5,  # "In perfect"
#     0.5, 0.5,  # "harmony"
#     0.5, 0.5,  # "Their hearts"
#     0.5, 0.5   # "are the same"
# ]
#
# play_notes_with_durations(notes_sequence, durations)

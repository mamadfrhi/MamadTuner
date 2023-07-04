
# Created at 21.06.2023 - iMamad Farrahi

import copy
from pyaudio import PyAudio, paInt16
from threading import Thread
import numpy as np
import sys
import tkinter as tk
import tkinter.font as tkFont


class AudioAnalyzer(Thread):
    # settings: (are tuned for best detecting string instruments like guitar)
    SAMPLING_RATE = 48000  # mac hardware: 44100, 48000, 96000
    CHUNK_SIZE = 1024  # number of samples
    BUFFER_TIMES = 50  # buffer length = CHUNK_SIZE * BUFFER_TIMES
    ZERO_PADDING = 3  # times the buffer length
    NUM_HPS = 3  # Harmonic Product Spectrum

    # overall frequency accuracy (step-size):  SAMPLING_RATE / (CHUNK_SIZE * BUFFER_TIMES * (1 + ZERO_PADDING)) Hz
    #               buffer length in seconds:  (CHUNK_SIZE * BUFFER_TIMES) / SAMPLING_RATE sec

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, queue, buffer=None, data_feed=None, *args, **kwargs):
        Thread.__init__(self, *args, **kwargs)

        self.queue = queue  # queue should be instance of ProtectedList (threading_helper.ProtectedList)
        self.buffer = np.zeros(self.CHUNK_SIZE * self.BUFFER_TIMES) if buffer is None else buffer
        self.hanning_window = np.hanning(len(self.buffer))
        self.running = False
        self.data_feed = data_feed

        try:
            self.audio_object = PyAudio()
            self.stream = self.audio_object.open(format=paInt16,
                                                 channels=1,
                                                 rate=self.SAMPLING_RATE,
                                                 input=True,
                                                 output=False,
                                                 frames_per_buffer=self.CHUNK_SIZE)
        except Exception as e:
            sys.stderr.write('Error: Line {} {} {}\n'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
            return

    @staticmethod
    def frequency_to_number(freq, a4_freq):
        """ converts a frequency to a note number (for example: A4 is 69)"""

        if freq == 0:
            sys.stderr.write("Error: No frequency data. Program has potentially no access to microphone\n")
            return 0

        return 12 * np.log2(freq / a4_freq) + 69

    @staticmethod
    def number_to_frequency(number, a4_freq):
        """ converts a note number (A4 is 69) back to a frequency """

        return a4_freq * 2.0**((number - 69) / 12.0)

    @staticmethod
    def number_to_note_name(number):
        """ converts a note number to a note name (for example: 69 returns 'A', 70 returns 'A#', ... ) """

        return AudioAnalyzer.NOTE_NAMES[int(round(number) % 12)]

    @staticmethod
    def frequency_to_note_name(frequency, a4_freq):
        """ converts frequency to note name (for example: 440 returns 'A') """

        number = AudioAnalyzer.frequency_to_number(frequency, a4_freq)
        note_name = AudioAnalyzer.number_to_note_name(number)
        return note_name

    @staticmethod
    def frequency_difference(current_freq, note_number):
        """ Calculate the difference between current frequency and ideal frequency """
        ideal_freq = AudioAnalyzer.number_to_frequency(note_number, 440)  # find ideal frequency
        deviation = 1200 * np.log2(
            current_freq / ideal_freq) if current_freq != 0 else 0  # Calculate deviation in cents
        return abs(current_freq - ideal_freq), deviation

    def run(self):
        """ Main function where the microphone buffer gets read and
            the fourier transformation gets applied """

        self.running = True

        while self.running:
            try:
                if self.data_feed is None:
                    # read microphone data
                    data = self.stream.read(self.CHUNK_SIZE, exception_on_overflow=False)
                    data = np.frombuffer(data, dtype=np.int16)

                    # append data to audio buffer
                    self.buffer[:-self.CHUNK_SIZE] = self.buffer[self.CHUNK_SIZE:]
                    self.buffer[-self.CHUNK_SIZE:] = data

                    # apply the fourier transformation on the whole buffer (with zero-padding + hanning window)
                    magnitude_data = abs(np.fft.fft(np.pad(self.buffer * self.hanning_window,
                                                           (0, len(self.buffer) * self.ZERO_PADDING),
                                                           "constant")))
                    # only use the first half of the fft output data
                    magnitude_data = magnitude_data[:int(len(magnitude_data) / 2)]

                    # HPS: multiply data by itself with different scalings (Harmonic Product Spectrum)
                    magnitude_data_orig = copy.deepcopy(magnitude_data)
                    for i in range(2, self.NUM_HPS+1, 1):
                        hps_len = int(np.ceil(len(magnitude_data) / i))
                        magnitude_data[:hps_len] *= magnitude_data_orig[::i]  # multiply every i element

                    # get the corresponding frequency array
                    frequencies = np.fft.fftfreq(int((len(magnitude_data) * 2) / 1),
                                                 1. / self.SAMPLING_RATE)

                    # set magnitude of all frequencies below 60Hz to zero
                    for i, freq in enumerate(frequencies):
                        if freq > 100:
                            magnitude_data[:i - 1] = 0
                            break

                    # put the frequency of the loudest tone into the queue
                    freq = round(frequencies[np.argmax(magnitude_data)], 2)
                    note_number = AudioAnalyzer.frequency_to_number(freq, 440)
                    note_number_rounded = round(note_number)  # round to nearest integer
                    freq_diff, deviation = self.frequency_difference(freq, note_number_rounded)
                    self.queue.put({
                        'frequency': freq,
                        'note_name': self.frequency_to_note_name(freq, 440),
                        'difference': freq_diff,
                        'deviation': deviation
                    })
                else:
                    data = self.data_feed

            except Exception as e:
                sys.stderr.write('Error: Line {} {} {}\n'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

        self.stream.stop_stream()
        self.stream.close()
        self.audio_object.terminate()

if __name__ == "__main__":
    from threading_helper import ProtectedList
    import time

    q = ProtectedList()
    a = AudioAnalyzer(q)
    a.start()

    window = tk.Tk()
    window.title('Music Tuner')
    window.geometry('400x200')  # width x height

    bigFontStyle = tkFont.Font(family="Lucida Grande", size=20)
    smallFontStyle = tkFont.Font(family="Lucida Grande", size=12)

    label_freq = tk.Label(window, text="", font=bigFontStyle)
    label_note = tk.Label(window, text="", font=bigFontStyle)
    label_diff = tk.Label(window, text="", font=smallFontStyle)
    label_dev = tk.Label(window, text="", font=smallFontStyle)

    label_freq.pack()
    label_note.pack()
    label_diff.pack()
    label_dev.pack()

    def update():
        q_data = q.get()
        if q_data is not None:
            diff = round(q_data['difference'], 3)
            dev = round(q_data['deviation'], 2)
            label_freq.config(text=f"Detected frequency: {round(q_data['frequency'], 2)} Hz")
            label_note.config(text=f"Closest note: {q_data['note_name']}")
            label_diff.config(text=f"Difference from ideal frequency: {diff} Hz")
            label_dev.config(text=f"Deviation: {dev} cents")

            if diff > 2:  # change the condition based on your requirements
                window.configure(background='red')
            else:
                window.configure(background='green')

        window.after(100, update)  # Update every 100 ms

    update()
    window.mainloop()

# ------------------------------------------------------------------------------
# -------------------------------- OFFLINE TEST --------------------------------
# --------------------- Comment above and uncomment below ----------------------
# ------------------------------------------------------------------------------
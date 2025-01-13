import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class SpectrogramGenerator:

    def __init__(self):
        pass

    def _normalization(self, data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude ** 2))
            s_norm[i] = data[i] / rms

        return s_norm

    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent
        spectrograms.

        INPUT:
            data: IQ samples (complex)

        RETURN:
            data_channel_ind_spec: Channel independent spectrograms.
        '''
        # Normalize the IQ samples.
        data = self._normalization(data)
        #print("data",np.array(data).shape)
        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256 * 0.4)  # Specify the row size
        num_column = int(np.floor((data.shape[1] - 256) / 128 + 1))  # Calculate columns

        # Initialize the spectrogram array
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec


    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.

        INPUT:
            sig: Complex IQ samples (time domain signal)

            win_len: Window length used in STFT.

            overlap: Overlap length used in STFT.

        RETURN:
            chan_ind_spec_amp: The generated channel independent spectrogram (magnitude).
        '''
        # Short-time Fourier transform (STFT).
        # 去除信号的直流分量
        sig = sig - np.mean(sig)

        f, t, spec = signal.stft(sig,
                                 window='hann',  # You can change window type here (e.g., 'hanning', 'hamming', etc.)
                                 nperseg=win_len,
                                 noverlap=overlap,
                                 nfft=win_len,
                                 return_onesided=False,
                                 padded=False,
                                 boundary=None)

        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)

        # Generate channel independent spectrogram (using amplitude of the FFT).
        chan_ind_spec_amp = np.abs(spec)

        # Optional: Logarithmic scaling (dB scale) if needed
        # chan_ind_spec_amp = np.log10(chan_ind_spec_amp + 1e-10)  # Avoid log(0)

        return chan_ind_spec_amp


    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        # Crop to 40% to 70% of the spectrogram's row range
        x_cropped = x[round(num_row * 0.3):round(num_row * 0.7)]

        return x_cropped

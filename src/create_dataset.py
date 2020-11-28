import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns


def load_audio(filename):
    # Load audio file
    audio_data, sampling_rate = librosa.load(filename, sr=None)
    dt = 1./sr
    time = np.linspace(0.0, len(audio_data)*dt, len(audio_data))
    return time, audio_data, sampling_rate


def compute_fft(audio_data, sr, n_bins):
    fft_binned = np.zeros(n_bins)
    freq_binned = np.zeros(n_bins)
    dt = 1./sr

    fourier = np.abs(np.fft.fft(audio_data))
    freq = np.fft.fftfreq(len(audio_data), d=dt)
    bs = np.argmax(freq) // n_bins
    for b in range(n_bins):
        fft_binned[b] = np.mean(fourier[b*bs: (b+1)*bs])
        freq_binned[b] = np.mean(freq[b*bs: (b+1)*bs])

    fft_binned /= np.max(fft_binned)
#    print(fourier.shape, freq_binned.shape, fft_binned.shape)

#    plt.plot(freq, fourier/np.max(fourier))
#    plt.plot(freq_binned, fft_binned/np.max(fft_binned))
#    plt.xlim([0, sr / 2])
#    plt.show()

    return freq_binned, fft_binned


def plot_waveform_fft(x, y, xlab, ylab, xlim, ylim, title, save, fs=14):
    plt.figure()

    subindex = len(y) * 100 + 11
    for i in range(len(y)):
        plt.subplot(subindex + i)
        plt.plot(x[i], y[i])

        plt.xlabel(xlab[i], fontsize=fs-1)
        plt.ylabel(ylab[i], fontsize=fs-1)
#        plt.xlim(xlim)
#        plt.ylim(ylim[i])
        if i == 0:
            plt.title(title, fontsize=fs)
        plt.grid('on')
    plt.tight_layout()
    plt.savefig(save)


if __name__ == '__main__':
    audio_path = '../../../Random/Kaggle/cats_dogs_audio/cats_dogs/'
    data_path = '../datafiles/dataset/'
    fig_path = '../figures/'

    cat = 'cat_'
    dog = 'dog_barking_'
    fex = '.wav'

    n_cat = 164
    n_dog = 113
    n_tot = n_cat + n_dog

    plot_cat = 10
    plot_dog = 6

    sr = 16000
    bin_size = 20
    n_bins = (sr//2) // bin_size

    data_set = np.zeros((n_tot, n_bins))
    y_targets = np.zeros((n_tot, 1))
    y_targets[167:] = 1  # 0 = cat, 1 = dog

    # Looping over all cat audio files
    for i in range(1, n_cat + 1):
        filename = audio_path + cat + str(i) + fex
        time, audio_data, sampling_rate = load_audio(filename)

        freq_binned, fft_binned = compute_fft(audio_data, sampling_rate, n_bins)
        data_set[i-1, :] = fft_binned[:]

        if i == plot_cat:
            plot_waveform_fft([time, freq_binned], [audio_data, fft_binned],
                              ['Time [s]', 'Frequency [Hz]'], ['Amplitude', 'Amplitude'],
                              [time[0], time[-1]], [freq_binned[0], freq_binned[-1]], 'Cat sample nr. %d' % i,
                              fig_path + 'cat_%d_nbins%d.png' % (i, n_bins))

    # Looping over all dog audio files
    for i in range(1, n_dog + 1):
        filename = audio_path + dog + str(i) + fex
        time, audio_data, sampling_rate = load_audio(filename)

        freq_binned, fft_binned = compute_fft(audio_data, sampling_rate, n_bins)
        data_set[n_cat + i - 1, :] = fft_binned[:]

        if i == plot_dog:
            plot_waveform_fft([time, freq_binned], [audio_data, fft_binned],
                              ['Time [s]', 'Frequency [Hz]'], ['Amplitude', 'Amplitude'],
                              [time[0], time[-1]], [freq_binned[0], freq_binned[-1]], 'Dog sample nr. %d' % i,
                              fig_path + 'dog_%d_nbins%d.png' % (i, n_bins))

    # Shuffle design matrix and targets
    shuffle = np.arange(len(y_targets))
    np.random.shuffle(shuffle)
    data_set = data_set[shuffle]
    y_targets = y_targets[shuffle]

    # Save data-set to file
    np.save(data_path + 'cat_dog_X_nbins%d' % n_bins, data_set)
    np.save(data_path + 'cat_dog_y_nbins%d' % n_bins, y_targets)

    plt.show()

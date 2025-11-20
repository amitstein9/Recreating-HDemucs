import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd

def calc_reconstruction_loss(separated_dir, mixture_file):
    """
    Calculate the reconstruction loss of separated audio files compared to the original mixture file.
    
    :param separated_dir: Directory containing separated audio files
    :param mixture_file: Path to the original mixture file
    :return: Reconstruction loss (in dB)
    """
    # Load the original mixture file
    mixture, sr = librosa.load(mixture_file, sr=None)

    # Iterate through separated audio files
    for audio_file in ['drums.wav', 'bass.wav', 'other.wav', 'vocals.wav']:
        separated_path = f"{separated_dir}/{audio_file}"
        separated, _ = librosa.load(separated_path, sr=sr)

        if mixture.shape[0] != separated.shape[0]:
            min_len = min(mixture.shape[0], separated.shape[0])
            print(f"mixture shape and separated shape are not equal: {mixture.shape[0]} != {separated.shape[0]}. setting both to {min_len}")
            mixture = mixture[:min_len]
            separated = separated[:min_len]
       
        mixture -= separated
        
    return np.sum(np.abs(mixture**2))/mixture.size

def compute_hnr(audio_file, time_step=0.01, minimum_pitch=75, energy_threshold=0, drums=False):
    """
    Compute the Harmonics-to-Noise Ratio (HNR) of an audio signal using the Praat method.
    
    :param audio_file: Path to the audio file (e.g., .wav file)
    :param time_step: Time step for analysis (default: 0.01s)
    :param minimum_pitch: Minimum pitch for analysis (default: 75 Hz)
    :return: Tuple of (time_values, hnr_values)
    """
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    y = librosa.to_mono(y)  # Convert to mono if stereo
    y = librosa.util.normalize(y)
    # print(f"Audio file: {audio_file}, Sample rate: {sr}, Duration: {len(y)/sr:.2f} seconds")
    # play_audio(y, sr)  # Play the audio file
    #compute the energy of the signal, making sure the length of the output vector is the same as the input signal
    if energy_threshold > 0 and energy_threshold < 1:
        energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
        energy = librosa.util.normalize(energy, axis=1)
        energy = np.ravel(energy)
        energy_expanded = np.repeat(energy, 512)
        energy_expanded = energy_expanded[:len(y)]
        y = y[energy_expanded > energy_threshold]

    # Compute the Harmonic-to-Percussive Ratio (HPR)
    harmonic, percussive = librosa.effects.hpss(y, margin=3.0)
    hpr = 10*np.log10(np.sum(np.abs(harmonic**2)) / np.sum(np.abs(percussive**2))) 
    # print(f"playing harmonic component")
    # play_audio(harmonic, sr)  # Play the harmonic component
    # print(f"playing harmonic component")
    # play_audio(percussive, sr)  # Play the percussive component
    # Compute the Harmonics-to-Noise Ratio (HNR) using Parselmouth
    

    #can create sound from full signal or from harmonic/percussive components. uncomment the line you want to use
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    # sound = parselmouth.Sound(percussive, sampling_frequency=sr) if drums else parselmouth.Sound(harmonic, sampling_frequency=sr)
    hnr = sound.to_harmonicity_cc(time_step=time_step, minimum_pitch=minimum_pitch, silence_threshold=energy_threshold)
    
    # Get the HNR values
    hnr_values = np.ravel(hnr.values)
    
    #clip HNR values to avoid numerical errors
    hnr_values[hnr_values < -50] = 0  # clip extremely negative values (numerical error)

    return np.mean(hnr_values), hpr

def plot_hnr(time_values, hnr_values):
    """Plot the HNR values over time."""
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, hnr_values, label='HNR (dB)', color='b')
    plt.xlabel("Time (s)")
    plt.ylabel("HNR (dB)")
    plt.title("Harmonics-to-Noise Ratio (HNR) over Time")
    plt.legend()
    plt.grid()
    plt.show()

def play_audio(y, sr):
    """Play the audio file."""
    sd.play(y[int(0.1*y.size):int(0.2*y.size)], samplerate=sr)
    sd.wait()  # Wait until the sound has finished playing

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute HNR from an audio file")
    parser.add_argument("separated_dir", type=str, help="Path to dir with separated audio files")
    parser.add_argument("mixture_file", type=str, help="Path to original mixture file")
    parser.add_argument("--calc_energy", type=float, default=0, help="if 0, ignore energy calc, else threshold signal by energy (default: 0.01s)")
    args = parser.parse_args()
    
    Drums_path = args.separated_dir + '\\drums.wav'
    Bass_path = args.separated_dir + '\\bass.wav'
    Others_path = args.separated_dir + '\\other.wav'
    Vocals_path = args.separated_dir + '\\vocals.wav'
    
    drums_hnr, drums_hpr = compute_hnr(Drums_path, energy_threshold=args.calc_energy, drums=True)
    bass_hnr, bass_hpr = compute_hnr(Bass_path, energy_threshold=args.calc_energy)
    # others_hnr, others_hpr = compute_hnr(Others_path)
    vocals_hnr, vocals_hpr = compute_hnr(Vocals_path, energy_threshold=args.calc_energy)

    reconstruction_loss = calc_reconstruction_loss(args.separated_dir, args.mixture_file)

    print(f"average HNR for drums: {(drums_hnr)}, HPR: {drums_hpr}")
    print(f"average HNR for bass: {(bass_hnr)} , HPR: {bass_hpr}")
    # print(f"average HNR for others: {(others_hnr)} , HPR: {others_hpr}")
    print(f"average HNR for vocals: {(vocals_hnr)} , HPR: {vocals_hpr}")
    print(f"Reconstruction loss: {reconstruction_loss}")
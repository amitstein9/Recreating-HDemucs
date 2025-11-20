#Songlist:
# test1: "Honolulu March" from "SpongeBob" - tv/folk - barely any Bass - EXCELLENT - Vocals+Drums quiet, Others really good, Bass fine
# test2: "Another One Bites the Dust" by Queen - rock - Vocal, Drums, Bass - GOOD
# test3: "Die House" from "Cuphead" - game/jazz - Vocal, Drums, Bass - GOOD - others is not the best
# test4: "la vie en rose" by Louis Armstrong - jazz - Vocal, Bass - ALMOST GOOD - Drums bled into "other" at end of track. otherwise excellent
# test5: "let's groove" by Earth, Wind & Fire - disco - Vocal, Drums, Bass - ALMOST GOOD - Vocoder confused the bass track and the others track, some bleeding
# test6: "Back in Black" by AC/DC - rock - Vocal, Drums, Bass - OK - Vocal with artifacts
# test7. "Uptown Funk" by Mark Ronson ft. Bruno Mars - pop - Vocal, Drums, Bass - OK - Bass fuzzy on low frequencies
# test8: "Lose Yourself" by Eminem - rap - Vocal, Drums - BAD - Bass exists even though no bass track, Drums with a lot of artifacts
# test9: "שני משוגעים" by עומר אדם - pop - Vocal, Drums, Bass - BAD - Bass, others muffled AND BLEEDS BADLY INTO VOCALS
# test10: "Shake It Off" by Taylor Swift - pop - Vocal, Drums, Bass - BAD - Bass

TESTNUMBERS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
import HNR
# import os
# import numpy as np
import matplotlib.pyplot as plt

BasePath = "C:\\Users\\Yuval\\Desktop\\עיבוד אודיו\\" #hard coded full path since it is simpler. change to your own path to run the script
SeparatedPath = BasePath + "\\demucs\\Separated\\htdemucs\\"
MixturePath = BasePath + "\\demucs\\"

DrumsHNR = []
DrumsHPR = []
BassHNR = []
BassHPR = []
VocalsHNR = []
VocalsHPR = []
ReconstructionLoss = []

for i in TESTNUMBERS:
    print(f"Processing test {i}...")
    SeparatedDir = SeparatedPath + f"test{i}\\"
    MixtureFile = MixturePath + f"test{i}.mp3"
    DrumsPath = SeparatedDir + "drums.wav"
    BassPath = SeparatedDir + "bass.wav"
    VocalsPath = SeparatedDir + "vocals.wav"

    drums_hnr, drums_hpr = HNR.compute_hnr(DrumsPath, energy_threshold=0.01, drums=True)
    bass_hnr, bass_hpr = HNR.compute_hnr(BassPath, energy_threshold=0.01)
    vocals_hnr, vocals_hpr = HNR.compute_hnr(VocalsPath, energy_threshold=0.01)

    reconstruction_loss = HNR.calc_reconstruction_loss(SeparatedDir, MixtureFile)

    DrumsHNR.append(drums_hnr)
    DrumsHPR.append(drums_hpr)
    BassHNR.append(bass_hnr)
    BassHPR.append(bass_hpr)
    VocalsHNR.append(vocals_hnr)
    VocalsHPR.append(vocals_hpr)
    ReconstructionLoss.append(reconstruction_loss)
print(f"done testing. plotting results...")
# Fig for each Plot
# HNR Values
plt.figure(figsize=(8, 5))
plt.plot(TESTNUMBERS, DrumsHNR, marker='o', label='Drums HNR')
plt.plot(TESTNUMBERS, BassHNR, marker='x', label='Bass HNR')
plt.plot(TESTNUMBERS, VocalsHNR, marker='*', label='Vocals HNR')
plt.title('HNR Values')
plt.xlabel('Test Number')
plt.ylabel('HNR (dB)')
plt.xticks(TESTNUMBERS)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# HPR Values
plt.figure(figsize=(8, 5))
plt.plot(TESTNUMBERS, DrumsHPR, marker='o', label='Drums HPR')
plt.plot(TESTNUMBERS, BassHPR, marker='x', label='Bass HPR')
plt.plot(TESTNUMBERS, VocalsHPR, marker='*', label='Vocals HPR')
plt.title('HPR Values')
plt.xlabel('Test Number')
plt.ylabel('HPR (dB)')
plt.xticks(TESTNUMBERS)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Reconstruction Loss
plt.figure(figsize=(8, 5))
plt.plot(TESTNUMBERS, ReconstructionLoss, marker='o', label='Reconstruction Loss')
plt.title('Reconstruction Loss')
plt.xlabel('Test Number')
plt.ylabel('Loss')
plt.xticks(TESTNUMBERS)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# One figure with all plots
# Plotting the results
# fig, axes = plt.subplots(3, 2, figsize=(12, 8))  # Create a 3x2 grid of subplots

# # First plot - HNR Values
# axes[0, 0].plot(TESTNUMBERS, DrumsHNR, marker='o', label='Drums HNR')
# axes[0, 0].plot(TESTNUMBERS, BassHNR, marker='x', label='Bass HNR')
# axes[0, 0].plot(TESTNUMBERS, VocalsHNR, marker='*', label='Vocals HNR')
# axes[0, 0].set_title('HNR Values')
# axes[0, 0].set_xlabel('Test Number')
# axes[0, 0].set_ylabel('HNR (dB)')
# axes[0, 0].set_xticks(TESTNUMBERS)
# axes[0, 0].legend()
# axes[0, 0].grid()

# # Second plot - HPR Values
# axes[0, 1].plot(TESTNUMBERS, DrumsHPR, marker='o', label='Drums HPR')
# axes[0, 1].plot(TESTNUMBERS, BassHPR, marker='x', label='Bass HPR')
# axes[0, 1].plot(TESTNUMBERS, VocalsHPR, marker='*', label='Vocals HPR')
# axes[0, 1].set_title('HPR Values')
# axes[0, 1].set_xlabel('Test Number')
# axes[0, 1].set_ylabel('HPR (dB)')
# axes[0, 1].set_xticks(TESTNUMBERS)
# axes[0, 1].legend()
# axes[0, 1].grid()

# # Third plot - Reconstruction Loss
# axes[1, 0].plot(TESTNUMBERS, ReconstructionLoss, marker='o', label='Reconstruction Loss')
# axes[1, 0].set_title('Reconstruction Loss')
# axes[1, 0].set_xlabel('Test Number')
# axes[1, 0].set_ylabel('Loss')
# axes[1, 0].set_xticks(TESTNUMBERS)
# axes[1, 0].legend()
# axes[1, 0].grid()

# # Remove unused subplots (since you specified a 3x2 layout but used only 3 plots)
# fig.delaxes(axes[1, 1])  # Remove empty subplot at (1,1)
# fig.delaxes(axes[2, 0])  # Remove empty subplot at (2,0)
# fig.delaxes(axes[2, 1])  # Remove empty subplot at (2,1)

# fig.tight_layout()  # Adjust layout to prevent overlap
# plt.show()
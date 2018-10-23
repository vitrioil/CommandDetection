import os
import scipy
import numpy as np
import librosa.display
from sample import Augment
import matplotlib.pyplot as plt

names = ["Ameya/", "Prem/", "Raj/", "Vinit/", "Vivek/"]
pos_list = [os.path.join("../data/Pos/",name) for name in names]
neg_list = [os.path.join("../data/Neg/"+name) for name in names]
bg_list = ["../data/BG/"]
func = ["pitch_speed", "distribution_noise"]
for pos in pos_list:
	print(os.listdir(pos))
	for f in os.listdir(pos):
		if "distribution" in f or "pitch" in f:
			continue
		rate, sample = scipy.io.wavfile.read(pos+f)
		for aug_f in func:
			new_sample = Augment.__dict__[aug_f](sample, rate)
			if new_sample is None:
				continue
			new_sample = np.clip(new_sample, 0, 65535)
			scipy.io.wavfile.write(filename=pos+f[:-4]+'_'+aug_f+".wav", data = new_sample.astype(np.int16), rate = rate)

for neg in neg_list:
	print(os.listdir(neg))
	for f in os.listdir(neg):
		rate, sample = scipy.io.wavfile.read(neg+f)
		for aug_f in func:
			new_sample = Augment.__dict__[aug_f](sample, rate)
			if new_sample is None:
				continue
			new_sample = np.clip(new_sample, 0, 65535)
			scipy.io.wavfile.write(filename=neg+f[:-4]+'_'+aug_f+".wav", data = new_sample.astype(np.int16), rate = rate)

for bg in bg_list[:1]:
	for f in os.listdir(bg):
		rate, sample = scipy.io.wavfile.read(bg+f)
		for aug_f in func:
			new_sample = Augment.__dict__[aug_f](sample, rate)
			if new_sample is None:
				continue
			new_sample = np.clip(new_sample, 0, 65535)
			#scipy.io.wavfile.write(filename=pos+f[:-4]+'_'+aug_f+".wav", data = new_sample, rate = rate)


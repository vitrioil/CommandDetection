import sys
import pydub
import numpy as np
from model import TModel
from sample import Sample
from pydub import AudioSegment
import wave, array

def make_stereo(file1, output):
    ifile = wave.open(file1)
    # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert comptype == 'NONE'  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel

    ofile = wave.open(output, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tostring())
    ofile.close()
if __name__ == "__main__":
	model = TModel(1999, 20, "dataset.h5", saved = True)
	model.detect_triggerword(int(sys.argv[2]), f"{sys.argv[1]}", plot_graph = True)

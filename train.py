import os
import sys
from model import TModel

if __name__ == "__main__":
	saved = True if sys.argv[1].lower() == "saved" else False
	model = TModel(1999, 20, "dataset.h5", saved = saved)
	model.train()

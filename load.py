import numpy as np
import nibabel as nib
import os

HGG = 'data/HGG/'
LGG = 'data/LGG/'

def normalize(img):
    mu = np.mean(img)
    sig = np.std(img)
    return (img - mu) / (5 * sig)

def load_data(path, nb=None):
    patients = sorted(os.listdir(path))
    print("Nb of patients:", len(patients))
    data = []
    segs = []
    for patient in patients:
        curr_path = os.path.join(path, patient)
        patient_data = sorted(os.listdir(curr_path))
        
        flair = normalize(np.transpose(nib.load(os.path.join(curr_path, patient_data[0])).get_fdata(), (2, 1, 0)))
        seg = normalize(np.transpose(nib.load(os.path.join(curr_path, patient_data[1])).get_fdata(), (2, 1, 0)))
        t1 = normalize(np.transpose(nib.load(os.path.join(curr_path, patient_data[2])).get_fdata(), (2, 1, 0)))
        t1ce = normalize(np.transpose(nib.load(os.path.join(curr_path, patient_data[3])).get_fdata(), (2, 1, 0)))
        t2 = normalize(np.transpose(nib.load(os.path.join(curr_path, patient_data[4])).get_fdata(), (2, 1, 0)))
        
        data.append(np.array([flair, t1, t1ce, t2]))
        segs.append(seg)
        
    return np.transpose(np.array(data), (0,2,3,4,1)), np.array(segs)


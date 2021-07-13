import numpy as np
import nibabel as nib
import os
from tempfile import mkdtemp
import os.path as path

# Changer avec vos path
HGG = '../brats_dataset/HGG/'
LGG = '../brats_dataset/LGG/'

def normalize(img):
    mu = np.mean(img)
    sig = np.std(img)
    return (img - mu) / (5 * sig)

def load_data(path, fp, start=0):
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

        segs.append(seg)
        if start % 10 == 0:
            print(start)

        fp[start] = np.transpose(np.asarray([flair, t1, t1ce, t2]), (1, 2, 3, 0))
        start += 1
    return fp, np.array(segs), len(patients)

# mkdtemp va créer un folder dans /tmp/
filename = path.join(mkdtemp(), 'newfile.dat')
print(filename)
fp = np.memmap(filename, dtype='float64', mode='w+', shape=(285, 155, 240, 240, 4))
fp, segs1, start = load_data(HGG, fp)
fp, segs2, _ = load_data(LGG, fp, start)


# Bus error entre les index 28 & 29
# newfp = np.memmap(filename, dtype='float64', mode='r', shape=(285, 155, 240, 240, 4))
# print(newfp.shape)
# print('29: ', newfp[29])
# print('284: ', newfp[284])

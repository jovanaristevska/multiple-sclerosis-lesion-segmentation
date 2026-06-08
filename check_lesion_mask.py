import nibabel as nib
import numpy as np

img = nib.load(r"D:\Documents\MANU\dataset_singletp\patient01\lesion.nii.gz")
data = img.get_fdata()
print("Unique values:", np.unique(data))
print("Shape:", data.shape)
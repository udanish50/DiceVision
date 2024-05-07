import torch
import cv2
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image import VisualInformationFidelity
from skimage.util import view_as_windows

folders = ['Orignal', 'Stable_Difusion', 'DELL2', 'Glide','DELL3']
desired_size = (299, 299)

def read_and_resize(folder, ext):
    images = {}
    for file in os.listdir(folder):
        if file.endswith(ext):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, desired_size)
            img_tensor = torch.tensor(img, dtype=torch.uint8)
            images[file.split('.')[0]] = img_tensor
    return images


originals = read_and_resize(folders[0], '.jpg')
stable_diffs = read_and_resize(folders[1], '.png')
dells2 = read_and_resize(folders[2], '.png')
glides = read_and_resize(folders[3], '.png')
dells3 = read_and_resize(folders[4], '.png')


folders = ['Orignal', 'Stable_Difusion', 'DELL2', 'Glide', 'DELL3']
desired_size = (299, 299)

def read_and_resize(folder, ext):
    images = []
    for file in os.listdir(folder):
        if file.endswith(ext):
            img = cv2.imread(os.path.join(folder, file))
            img = cv2.resize(img, desired_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            images.append(img)
    images_tensor = torch.tensor(images, dtype=torch.uint8)
    images_tensor = images_tensor.permute(0, 3, 1, 2)
    return images_tensor

def compute_fid_with_originals(originals, compared_set):
    fid = FrechetInceptionDistance(feature=64)
    fid.update(originals, real=True)
    fid.update(compared_set, real=False)
    return fid.compute()

def compute_kid_with_originals(originals, compared_set):
    kid = KernelInceptionDistance(subsets=3, subset_size=3)  # Adjust as needed
    kid.update(originals, real=True)
    kid.update(compared_set, real=False)
    return kid.compute()

def compute_inception_score(imgs):
    inception = InceptionScore()
    inception.update(imgs)
    return inception.compute()

def compute_lpips(img1, img2, net_type='alex'):
    img1_normalized = img1.float() / 127.5 - 1
    img2_normalized = img2.float() / 127.5 - 1
    lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, reduction='mean')
    return lpips(img1_normalized, img2_normalized)

def compute_ssim(img1, img2, data_range=255):
    ssim = StructuralSimilarityIndexMeasure(data_range=data_range)
    return ssim(img1.float(), img2.float())

def compute_psnr(img1, img2, data_range=None):
    psnr = PeakSignalNoiseRatio(data_range=data_range)
    return psnr(img1.float(), img2.float())

def compute_ms_ssim(img1, img2, data_range=255):
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=data_range)
    return ms_ssim(img1.float(), img2.float())

def compute_mifid(real_imgs, fake_imgs, feature=64, cosine_distance_eps=0.1):
    mifid = MemorizationInformedFrechetInceptionDistance(
        feature=feature, 
        normalize=False, 
        cosine_distance_eps=cosine_distance_eps
    )
    mifid.update(real_imgs, real=True)
    mifid.update(fake_imgs, real=False)
    return mifid.compute()

def compute_hfid(originals, generated, weight_ds=0.5, weight_ps=0.5):
    fid = compute_fid_with_originals(originals, generated)    
    lpips_scores = []
    for original_img, generated_img in zip(originals, generated):
        lpips_score = compute_lpips(original_img.unsqueeze(0), generated_img.unsqueeze(0))
        lpips_scores.append(lpips_score)
    avg_lpips = torch.mean(torch.tensor(lpips_scores))
    hfid = weight_ds * fid + weight_ps * avg_lpips
    return hfid.item()

originals = read_and_resize(folders[0], '.jpg')
stable_diffs = read_and_resize(folders[1], '.png')
dells2 = read_and_resize(folders[2], '.png')
glides = read_and_resize(folders[3], '.png')
dells3 = read_and_resize(folders[4], '.png')

fid_scores, kid_scores, inception_scores = {}, {}, {}
datasets = {
    "Stable Diffusions": stable_diffs,
    "DELL2": dells2,
    "Glide": glides,
    "DELL3": dells3,
}

for name, dataset in datasets.items():
    fid_scores[name] = compute_fid_with_originals(originals, dataset)
    kid_scores[name] = compute_kid_with_originals(originals, dataset)
    inception_scores[name] = compute_inception_score(dataset)

print("\nFID scores compared to Originals:")
for name, score in fid_scores.items():
    print(f"{name}: {score}")

print("\nKID scores compared to Originals:")
for name, score in kid_scores.items():
    print(f"{name}: {score}")

print("\nInception Scores:")
for name, score in inception_scores.items():
    is_mean, is_std = score
    print(f"{name}: Mean = {is_mean}, Std = {is_std}")

lpips_scores = {
    "Stable Diffusions": compute_lpips(originals, stable_diffs),
    "DELL2": compute_lpips(originals, dells2),
    "Glide": compute_lpips(originals, glides),
    "DELL3": compute_lpips(originals, dells3),
}

# Print LPIPS scores
print("\nLPIPS scores compared to Originals:")
for name, score in lpips_scores.items():
    print(f"{name}: {score.item()}")

ssim_scores = {
    "Stable Diffusions": compute_ssim(originals, stable_diffs),
    "DELL2": compute_ssim(originals, dells2),
    "Glide": compute_ssim(originals, glides),
    "DELL3": compute_ssim(originals, dells3),
}

# Print SSIM scores
print("\nSSIM scores compared to Originals:")
for name, score in ssim_scores.items():
    print(f"{name}: {score.item()}")

# Compute MS-SSIM scores
ms_ssim_scores = {
    "Stable Diffusions": compute_ms_ssim(originals, stable_diffs),
    "DELL2": compute_ms_ssim(originals, dells2),
    "Glide": compute_ms_ssim(originals, glides),
    "DELL3": compute_ms_ssim(originals, dells3),
}

# Print MS-SSIM scores
print("\nMS-SSIM scores compared to Originals:")
for name, score in ms_ssim_scores.items():
    print(f"{name}: {score.item()}")

# Compute PSNR scores
psnr_scores = {
    "Stable Diffusions": compute_psnr(originals, stable_diffs),
    "DELL2": compute_psnr(originals, dells2),
    "Glide": compute_psnr(originals, glides),
    "DELL3": compute_psnr(originals, dells3),
}

# Print PSNR scores
print("\nPSNR scores compared to Originals:")
for name, score in psnr_scores.items():
    print(f"{name}: {score.item()}")

# Compute MIFID scores
mifid_scores = {
    "Stable Diffusions": compute_mifid(originals, stable_diffs),
    "DELL2": compute_mifid(originals, dells2),
    "Glide": compute_mifid(originals, glides),
    "DELL3": compute_mifid(originals, dells3),
}

# Print MIFID scores
print("\nMIFID scores compared to Originals:")
for name, score in mifid_scores.items():
    print(f"{name}: {score.item()}")

hfid_scores = {
    "Stable Diffusions": compute_hfid(originals, stable_diffs),
    "DELL2": compute_hfid(originals, dells2),
    "Glide": compute_hfid(originals, glides),
    "DELL3": compute_hfid(originals, dells3),
}

# Print MIFID scores
print("\nHFID scores compared to Originals:")
for name, score in hfid_scores.items():
    print(f"{name}: {score}")

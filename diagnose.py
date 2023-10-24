import joblib
import numpy as np
import radiomics
import SimpleITK as sitk
import torch
from radiomics import featureextractor
from unet import UNet

radiomics.setVerbosity(40)


def threshold(data: torch.Tensor, level: float = 0.5) -> torch.Tensor:
    scaled = (data - data.min()) / (data.max() - data.min())
    scaled[scaled < level] = 0
    scaled[scaled >= level] = 1
    return scaled


def segment(image: np.ndarray) -> np.ndarray:
    model = UNet(residual=False, cat=False)
    model.load_state_dict(torch.load(r".\models\unet_final.pt"))

    pred = threshold(model(torch.tensor([[image]], dtype=torch.float32)))
    return pred.cpu().detach().numpy()


def get_haralick_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(image)
    msk = sitk.GetImageFromArray(mask)
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName("glcm")
    result = extractor.execute(img, msk)
    return np.array([value for key, value in result.items() if "glcm" in key])


def classify_tumor(x: np.ndarray) -> bool:
    model = joblib.load(r".\models\svc.joblib")
    return bool(model.predict(x.reshape(1, -1)))

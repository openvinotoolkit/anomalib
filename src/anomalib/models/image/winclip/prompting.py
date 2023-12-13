

NORMAL_STATES = [
    "{}",
    "flawless {}",
    "perfect {}",
    "unblemished {}",
    "{} without flaw",
    "{} without defect",
    "{} without damage"
]

ANOMALOUS_STATES = [
    "damaged {}",
    "{} with flaw",
    "{} with defect",
    "{} with damage"
]

TEMPLATES = [
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection."
]

def create_prompt_ensemble(object_name: str):
    """Create prompt ensemble for WinCLIP.
    
    Args:
        class_name (str): Name of the object.
    """
    normal_states = [state.format(object_name) for state in NORMAL_STATES]
    normal_ensemble = [template.format(state) for state in normal_states for template in TEMPLATES]

    anomalous_states = [state.format(object_name) for state in ANOMALOUS_STATES]
    anomalous_ensemble = [template.format(state) for state in anomalous_states for template in TEMPLATES]
    return normal_ensemble, anomalous_ensemble

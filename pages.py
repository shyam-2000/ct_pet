import diagnose
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

# from streamlit_modal import Modal


@st.cache_resource
def load_image(image_file: str) -> np.ndarray:
    return np.load(image_file)


def get_box(*args, **kwargs):
    return {
        "top": st.session_state.get("top", 224),
        "left": st.session_state.get("left", 224),
        "width": 64,
        "height": 64,
    }


def login_page():
    # modal = Modal("Data and Security", "privacy_policy_modal")
    # modal = None
    st.title("Login")
    st.text_input("Username")
    st.text_input("Password", type="password")
    open_modal = st.button(label="Read")
    if open_modal:
        with st.container():
            st.write("Personal data shall be:")
            st.markdown(
                """
                1. processed lawfully, fairly and in a transparent manner in relation to individuals (lawfulness, fairness and transparency)
                1. collected for specified, explicit and legitimate purposes and not further processed in a manner that is incompatible with those purposes; further processing for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes shall not be considered to be incompatible with the initial purposes (purpose limitation)
                1. adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed (data minimisation)
                1. accurate and, where necessary, kept up to date; every reasonable step must be taken to ensure that personal data that are inaccurate, having regard to the purposes for which they are processed, are erased or rectified without delay (accuracy)
                1. kept in a form which permits identification of data subjects for no longer than is necessary for the purposes for which the personal data are processed; personal data may be stored for longer periods insofar as the personal data will be processed solely for archiving purposes in the public interest, scientific or historical research purposes or statistical purposes subject to implementation of the appropriate technical and organisational measures required by the GDPR in order to safeguard the rights and freedoms of individuals (storage limitation)
                1. processed in a manner that ensures appropriate security of the personal data, including protection against unauthorised or unlawful processing and against accidental loss, destruction or damage, using appropriate technical or organisational measures (integrity and confidentiality).
                """
            )
    if st.checkbox("Agree to the privacy policy") and st.button("Login"):
        st.session_state["page"] = "upload"
        st.experimental_rerun()


def upload_page():
    st.title("Radiogram")
    image_file = st.file_uploader("Upload the CT scan", type="npy")
    if "image" in st.session_state:
        image_file = st.session_state["image"]

    if image_file:
        st.header("Select region of interest")
        ct_scan = load_image(image_file)
        if "slice" not in st.session_state:
            # st.session_state["slice"] = ct_scan.shape[0] // 2
            st.session_state["slice"] = 140
        slc = st.session_state["slice"]

        cropped: dict[str, int] = st_cropper(
            Image.fromarray(ct_scan[slc, :, :]),
            aspect_ratio=(1, 1),
            box_algorithm=get_box,
            return_type="box",
        )

        buttons = {
            r"$\triangleleft\triangleleft$": -5,
            r"$\triangleleft$": -1,
            r"$\triangleright$": 1,
            r"$\triangleright\triangleright$": 5,
        }
        for button, (icon, offset) in zip(st.columns(4), buttons.items()):
            with button:
                if st.button(icon, key=f"button_{icon}"):
                    st.session_state["top"] = cropped["top"]
                    st.session_state["left"] = cropped["left"]
                    st.session_state["slice"] = slc + offset
                    st.experimental_rerun()

        st.write("Preview")
        st.write(slc)
        st.image(
            ct_scan[
                slc,
                cropped["top"] : cropped["top"] + cropped["height"],
                cropped["left"] : cropped["left"] + cropped["width"],
            ],
            clamp=True,
        )

        if st.button("Find tumor"):
            top = cropped["top"]
            left = cropped["left"]
            np.save(
                r".\data\tmp\scan.npy",
                ct_scan[slc : slc + 16, top : top + 64, left : left + 64],
            )

            st.session_state["page"] = "segmentation"
            st.session_state["image"] = image_file
            st.experimental_rerun()


def segmentation_page():
    st.title("Segmented mask")
    ct_scan = np.load(r".\data\tmp\scan.npy")
    mask = diagnose.segment(ct_scan)
    original, prediction = st.columns(2)

    with original:
        for i, col in enumerate(st.columns(4)):
            col.image(ct_scan[i, :, :], clamp=True)
            col.image(ct_scan[i + 4, :, :], clamp=True)
            col.image(ct_scan[i + 8, :, :], clamp=True)
            col.image(ct_scan[i + 12, :, :], clamp=True)

    with prediction:
        for i, col in enumerate(st.columns(4)):
            col.image(mask[0, 0, i, :, :], clamp=True)
            col.image(mask[0, 0, i + 4, :, :], clamp=True)
            col.image(mask[0, 0, i + 8, :, :], clamp=True)
            col.image(mask[0, 0, i + 12, :, :], clamp=True)

    if st.button("Go back"):
        st.session_state["page"] = "upload"
        st.experimental_rerun()

    if st.button("Accept"):
        np.save(r".\data\tmp\mask.npy", mask[0, 0, :, :, :])
        st.session_state["page"] = "diagnosis"
        st.experimental_rerun()


def diagnosis_page():
    st.title("Diagnosis")
    image = np.load(r".\data\tmp\scan.npy")
    mask = np.load(r".\data\tmp\mask.npy")
    features = diagnose.get_haralick_features(image, mask)
    result = diagnose.classify_tumor(features)
    diagnosis = "Malignant" if result else "Benign"
    st.write(f"Our algorithm has detected a {diagnosis} tumor")
    if st.button("New diagnosis"):
        st.session_state.clear()
        st.session_state["page"] = "upload"
        st.experimental_rerun()

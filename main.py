import pages
import streamlit as st


def main():
    st.set_page_config(layout="centered")
    # app_mode = st.sidebar.radio(
    #     "Select Page", ["Home Page", "Prediction", "privacy policy"]
    # )

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if st.session_state["page"] == "login":
        pages.login_page()

    if st.session_state["page"] == "upload":
        pages.upload_page()

    if st.session_state["page"] == "segmentation":
        pages.segmentation_page()

    if st.session_state["page"] == "diagnosis":
        pages.diagnosis_page()


if __name__ == "__main__":
    main()

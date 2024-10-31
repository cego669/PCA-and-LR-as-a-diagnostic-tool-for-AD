import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import joblib as jb
from sklearn.preprocessing import StandardScaler, minmax_scale
import datetime
from sklearn.decomposition import PCA
import os
import json



############################################################### USEFUL FUNCTIONS

def generate_report(patient_name, patient_sex, patient_dob, exam_loc, patient_img_path):

    patient_vector = np.array([nib.load(patient_img_path).get_fdata()])
    patient_vector = patient_vector.reshape((patient_vector.shape[0], np.prod(patient_vector.shape[1:])))
    patient_vectorM = patient_vector[0, mask != 0]
    patient_vectorMS = minmax_scale(patient_vectorM).reshape((1, -1))

    score = clf_intercept + np.sum(pca.transform(patient_vectorMS) * clf_coef, axis = 1)

    ########## PLOT
    # defining StandardScaler to scale log odds over the non-Alzheimer disease group
    scaler = StandardScaler()

    # scaling log odds over the non-Alzheimer disease group
    scaler.fit(train_log_odds[np.array(y_train) == 0].reshape(-1, 1))

    # applying the standard scaler object over all the train and test log odds
    train_log_oddsS = scaler.transform(train_log_odds.reshape(-1, 1))
    score = scaler.transform(score.reshape((-1, 1)))

    # calculating the cut-off (50% prob for AD) line
    cutoff = (np.log(.5/(1 - .5)) - scaler.mean_)/np.sqrt(scaler.var_)

    # defining parameters for the plot
    np.random.seed(670)
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1, 2, figsize = (12, 13))

    ######################################### TRAINING DATA
    # plotting points for the non-Alzheimer disease group
    ax[0].scatter(np.random.normal(0, 1, len(train_log_oddsS[np.array(y_train) == 0])), train_log_oddsS[np.array(y_train) == 0],
            edgecolors = "k", color = "cyan")
    ax[0].scatter(x = 5, y = 0, color = "cyan", s = 100)
    ax[0].errorbar(x = 5, y = 0, yerr = 1, color = "cyan", capsize = 5,
                elinewidth = 4, capthick = 4)

    # plotting points for the Alzheimer disease group
    ax[0].scatter(np.random.normal(10, 1, len(train_log_oddsS[np.array(y_train) == 1])), train_log_oddsS[np.array(y_train) == 1],
            edgecolors = "k", color = "magenta")
    ax[0].scatter(x = 15, y = np.mean(train_log_oddsS[np.array(y_train) == 1]),
                color = "magenta", s = 100)
    ax[0].errorbar(x = 15, y = np.mean(train_log_oddsS[np.array(y_train) == 1]), yerr = np.std(train_log_oddsS[np.array(y_train) == 1]),
                color = "magenta", capsize = 5, elinewidth = 4, capthick = 4)

    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    ax[0].set_xticks([0, 10])
    ax[0].set_xticklabels(["nAD", "AD"])

    ax[0].hlines(y = cutoff, linewidth = 2, linestyles = "--", xmin = -2, xmax = 17)
    ax[0].set_ylabel("Output score")

    ax[0].set_title("Reference dataset (ADNI)")

    ######################################### PATIENT'S SCORE
    ax[1].scatter(0, score[0][0],
            edgecolors = "k", color = "teal", label = "Patient's score: {:.2f}".format(score[0][0]))

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].set_yticks([])

    ax[1].set_xticks([0])
    ax[1].set_xticklabels(["Patient"])

    ax[1].hlines(y = cutoff, linewidth = 2, linestyles = "--", xmin = -2, xmax = 17,
                label = "50% probability line: 2.09")

    ax[1].set_title("Patient's score")

    # adjusting y lim
    lim_left = ax[0].get_ylim()
    lim_right = ax[1].get_ylim()
    ax[0].set_ylim((min(lim_left[0], lim_right[0]), max(lim_left[1], lim_right[1])))
    ax[1].set_ylim((min(lim_left[0], lim_right[0]), max(lim_left[1], lim_right[1])))
    ax[1].set_xlim((-1, 1))

    ##############################################################################
    # Brief description
    plt.figtext(0.1, 0, """
    nAD: non-Alzheimer disease group; AD: Alzheimer disease group.
    Validation metrics (CDI, n = 100 nAD / 92 AD):  Sensitivity = 89.13%, Precision = 86.32%, Accuracy = 88.02%
    Reference: Gonçalves de Oliveira CE, Araújo WM, Teixeira ABMJ, Gonçalves GL, Itikawa EN. PCA and logistic regression in
    2-[18F]FDG PET neuroimaging as an interpretable and diagnostic tool for Alzheimer's disease. Phys Med Biol.
    2023 Nov 17. doi: 10.1088/1361-6560/ad0ddd. PMID: 37976549.""", fontsize = 12)

    # Brief description 2
    plt.figtext(0.18, .93, """by implementing Machine Learning techquines such as Principal Component Analysis and Logistic Regression""", fontsize = 12)
    #############################################################################

    plt.figtext(.1, 1.03, f"""
    Name: {patient_name}
    Date of birth: {patient_dob}
    Sex: {patient_sex}
    Exam location: {exam_loc}""", fontsize = 15)

    ##############################################################################
    plt.suptitle("Alzheimer's disease prediction", fontsize = 20, fontweight="bold")
    ax[1].legend()
    plt.savefig("{}_AD_report.png".format(patient_name.replace(" ", "_")), dpi = 650, bbox_inches = "tight")


# load "fragmented" pca
def load_fragmented_pca(directory='fragmented_pca'):
    """
    Loads PCA attributes from fragmented files.
    
    Parameters:
    - directory: Directory where the fragments are saved.
    
    Returns:
    - Reconstructed PCA object.
    """
    # Load PCA parameters
    with open(os.path.join(directory, 'pca_params.json'), 'r') as f:
        pca_params = json.load(f)
    
    # Create a new PCA instance with the loaded parameters
    pca = PCA(
        n_components=pca_params['n_components'],
        svd_solver=pca_params['svd_solver'],
        tol=pca_params['tol'],
        copy=pca_params['copy'],
        whiten=pca_params['whiten'],
        random_state=pca_params['random_state']
    )
    
    # Load and set attributes
    pca.explained_variance_ = np.load(os.path.join(directory, 'explained_variance_.npy'))
    pca.explained_variance_ratio_ = np.load(os.path.join(directory, 'explained_variance_ratio_.npy'))
    pca.singular_values_ = np.load(os.path.join(directory, 'singular_values_.npy'))
    pca.mean_ = np.load(os.path.join(directory, 'mean_.npy'))
    
    # Load and reconstruct `components_`
    components_parts = pca_params.get('components_parts', 5)
    components_list = []
    for i in range(1, components_parts + 1):
        part = np.load(os.path.join(directory, f'components_part_{i}.npy'))
        components_list.append(part)
    
    pca.components_ = np.concatenate(components_list, axis=1)
    
    return pca


############################################################### LOADING USEFUL FILES

# Loading model parameters
clf_coef = jb.load("model/clf_coef.pkl")
clf_intercept = jb.load("model/clf_intercept.pkl")
pca = load_fragmented_pca(directory='model/fragmented_pca')

# Loading mask
mask = jb.load("data/mask/mask.pkl")

# Loading only y_train and then pre calculated train_log_odds
y_train = jb.load("model/y_train_only.pkl")
train_log_odds = jb.load("model/train_log_odds.pkl")


############################################################### SIDE BAR
section = st.sidebar.selectbox("Section:", ["Generate Report"])

# texto da aba
st.sidebar.markdown("""""", unsafe_allow_html=True)

###################################### SECTION: GENERATE REPORT
if section == "Generate Report":

    st.markdown("""# Alzheimer's disease prediction""")

    with st.form(key = "user_input"):
        
        st.markdown("""
        <h5 style='text-align: center;'>Patient's Data</h5>
        """, unsafe_allow_html=True)

        patient_name = st.text_input("Name")
        patient_sex = st.selectbox("Sex", ["Female", "Male"])
        patient_dob = st.date_input("Date of Birth",
                                    min_value=datetime.date(1900, 1, 1),
                                    max_value=datetime.datetime.now())
        exam_loc = st.text_input("Exam location")
        
        st.markdown("""
        <h5 style='text-align: center;'>Image file (.img/.hdr)</h5>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader(".IMG FILE",
                                        type=["img"])
        
        uploaded_hdr = st.file_uploader(".HDR FILE",
                                        type=["hdr"])

        predict = st.columns(5)[-1].form_submit_button("Get Report")

    if predict:

        if uploaded_img is not None and uploaded_hdr is not None:
            # saving files temporarily
            with open("image_file.img", "wb") as f:
                f.write(uploaded_img.getbuffer())

            with open("image_file.hdr", "wb") as f:
                f.write(uploaded_hdr.getbuffer())

            generate_report(patient_name, patient_sex, patient_dob, exam_loc, "image_file.img")
            st.image("{}_AD_report.png".format(patient_name.replace(" ", "_")))

            st.download_button("Download report",
                               data=open("{}_AD_report.png".format(patient_name.replace(" ", "_")), "rb"),
                               file_name="{}_AD_report.png".format(patient_name.replace(" ", "_")))
        else:
            st.error("Please, upload the image file!")
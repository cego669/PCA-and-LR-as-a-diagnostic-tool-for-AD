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
st.sidebar.image("aimip_logo.png", use_column_width=True)

section = st.sidebar.selectbox("Section:", ["Generate Report", "Explore Neuroimage", "Understanding the Model"])

# texto da aba
st.sidebar.markdown('''
This web application allows you to use the machine learning model implemented for the scientific article ["PCA and logistic regression in 2-[18F]FDG PET neuroimaging as an interpretable and diagnostic tool for Alzheimer's disease"](https://iopscience.iop.org/article/10.1088/1361-6560/ad0ddd) and interact with PET-FDG neuroimages in NIFTI format.

Please note that the model outputs should not be used or interpreted without the supervision of appropriately specialized nuclear physicians. Therefore, **the model is unable to make clinical decisions alone**.

Our work can be cited as:

```
Gonçalves de Oliveira CE, Araújo WM, Teixeira ABMJ, Gonçalves GL,
Itikawa EN. PCA and logistic regression in 2-[18F]FDG PET neuroimaging as
an interpretable and diagnostic tool for Alzheimer's disease. Phys Med Biol.
2023 Nov 17. doi: 10.1088/1361-6560/ad0ddd. PMID: 37976549.
```

Lead author:
                                        
[Carlos Eduardo Gonçalves de Oliveira](https://www.linkedin.com/in/cego669/) ([@cego669](https://github.com/cego669))
''', unsafe_allow_html=True)

###################################### SECTION: GENERATE REPORT
if section == "Generate Report":

    st.markdown(
        """
        <h1 style='text-align: center;'>Alzheimer's disease prediction</h1>
        """,
        unsafe_allow_html=True
    )

    with st.form(key = "user_input"):
        
        st.markdown("""
        <h5 style='text-align: center;'>📋 Patient's Data</h5>
        """, unsafe_allow_html=True)

        patient_name = st.text_input("Name")
        patient_sex = st.selectbox("Sex", ["Female", "Male"])
        patient_dob = st.date_input("Date of Birth",
                                    min_value=datetime.date(1900, 1, 1),
                                    max_value=datetime.datetime.now())
        exam_loc = st.text_input("Exam location")
        
        st.markdown("""
        <h5 style='text-align: center;'>🧠 PET-FDG neuroimage file (.img/.hdr)</h5>
        
        Please, note that the neuroimage must be dully reoriented (if necessary), spatially normalized (MNI space) and smoothed (default settings) using the tools available in [SPM12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/).""", unsafe_allow_html=True)

        uploaded_img = st.file_uploader(".img file",
                                        type=["img"])
        
        uploaded_hdr = st.file_uploader(".hdr file",
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

###################################### SECTION: EXPLORE NEUROIMAGE
if section == "Explore Neuroimage":
    st.markdown(
        """
        <h1 style='text-align: center;'>Explore Neuroimage</h1>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("Upload a `.img` and `.hdr` file to explore the PET-FDG neuroimage interactively across all three planes.")
    
    uploaded_img = st.file_uploader(".img file", type=["img"])
    uploaded_hdr = st.file_uploader(".hdr file", type=["hdr"])
    
    if uploaded_img is not None and uploaded_hdr is not None:
        with st.spinner("Processing neuroimage... Please wait."):
            # Save files temporarily
            with open("temp_image.img", "wb") as f:
                f.write(uploaded_img.getbuffer())
            with open("temp_image.hdr", "wb") as f:
                f.write(uploaded_hdr.getbuffer())
            
            # Load neuroimage
            img = nib.load("temp_image.img")
            data = img.get_fdata()
            
            st.markdown("---")
            st.markdown(
                """
                <h2 style='text-align: center;'>Visualization Pane</h2>
                """,
                unsafe_allow_html=True
            )

            # Create columns for side-by-side layout
            col1, col2, col3 = st.columns(3)

            # Axial plane
            with col1:
                #st.markdown("### Axial Plane")
                axial_slice = st.slider("Axial Plane Slice:", 0, data.shape[2] - 1, data.shape[2] // 2, key="axial_slider")
                axial_data = data[:, :, axial_slice]
                axial_aspect_ratio = axial_data.shape[0] / axial_data.shape[1]
                fig_axial = px.imshow(
                    axial_data,
                    color_continuous_scale="gray",
                    aspect=0.835
                )
                fig_axial.update_layout(
                    height=320,
                    width=320,
                    coloraxis_showscale=False  # Remove color bar
                )
                st.plotly_chart(fig_axial, use_container_width=False)
            
            # Coronal plane
            with col2:
                #st.markdown("### Coronal Plane")
                coronal_slice = st.slider("Coronal Plane Slice:", 0, data.shape[1] - 1, data.shape[1] // 2, key="coronal_slider")
                coronal_data = data[:, coronal_slice, :]
                coronal_aspect_ratio = coronal_data.shape[0] / coronal_data.shape[1]
                fig_coronal = px.imshow(
                    coronal_data,
                    color_continuous_scale="gray",
                    aspect=1.0
                )
                fig_coronal.update_layout(
                    height=320,
                    width=320,
                    coloraxis_showscale=False  # Remove color bar
                )
                st.plotly_chart(fig_coronal, use_container_width=False)
            
            # Sagittal plane
            with col3:
                #st.markdown("### Sagittal Plane")
                sagittal_slice = st.slider("Sagittal Plane Slice:", 0, data.shape[0] - 1, data.shape[0] // 2, key="sagittal_slider")
                sagittal_data = data[sagittal_slice, :, :]
                sagittal_aspect_ratio = sagittal_data.shape[0] / sagittal_data.shape[1]
                fig_sagittal = px.imshow(
                    sagittal_data,
                    color_continuous_scale="gray",
                    aspect=0.835
                )
                fig_sagittal.update_layout(
                    height=320,
                    width=320,
                    coloraxis_showscale=False  # Remove color bar
                )
                st.plotly_chart(fig_sagittal, use_container_width=False)
        
    else:
        st.warning("Please upload both `.img` and `.hdr` files.")

if section == "Understanding the Model":
    st.markdown("""
        <h1 style='text-align: center;'>Understanding the Model</h1>
        """, unsafe_allow_html=True)

    st.markdown("""
        **How the model works?**

        Our Alzheimer's disease diagnostic model combines two important machine learning techniques: 
        **Principal Component Analysis (PCA)** and **Logistic Regression (LR)**.

        ### 1. Principal Component Analysis (PCA)
        PCA is a dimensionality reduction technique that simplifies a dataset while preserving the most important information. Imagine having several brain images where each voxel is a variable. PCA transforms the data so that:

        - **Principal Components** are calculated as linear combinations of the original variables (voxels). In this sense, the principal components capture patterns over correlated voxels.
        - The **First Principal Component** captures the most variance in the data, followed by the second, and so on.

        For neuroimages, PCA allows us to focus on the most relevant patterns, reducing complexity without losing essential data.

        ### 2. Logistic Regression (LR)
        After extracting the principal components, we use LR to classify patients:

        - Each principal component is assigned a weight indicating its influence on the diagnosis.
        - LR calculates the probability of a patient belonging to the Alzheimer's group (AD) or the non-Alzheimer's group (nAD) using the formula:
    """, unsafe_allow_html=True)

    st.latex(r"P(\text{AD}) = \frac{1}{1 + e^{-\left(\beta_0 + \beta_1 \cdot PC_1 + \beta_2 \cdot PC_2 + \dots + \beta_p \cdot PC_p\right)}}")

    st.markdown("""
    Here:
    """, unsafe_allow_html=True)

    st.latex(r"\beta_0 \text{ is the intercept.}")

    st.latex(r"\beta_1, \beta_2, \dots, \beta_p \text{ are the weights of the principal components.}")

    st.markdown("""
    ### 3. Regularization for Better Generalization
    To prevent the model from overfitting the data, we use a technique called **regularization**:

    - **L1 Regularization (LASSO):** Reduces less important coefficients to zero, simplifying the model.
    - **L2 Regularization ():** less important coefficients also tend to zero, but in a less 'agressive' way.
    - This ensures that only the most significant patterns contribute to the diagnosis.

    ### Model Interpretation
    One advantage of this model is its **interpretability**:

    - Brain regions highlighted in red on the figure below represent a high probability of Alzheimer's, while blue regions indicate a low probability.
    - This approach allows us to understand how the model makes decisions, which is crucial for medical applications.
    """, unsafe_allow_html=True)

    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("figure4.png", width=300, caption="Resulting linear combination of principal components adjusted by their respective weights in a sequence of axial slices. Regions in red are associated with a higher probability for AD given high intensities in the voxels while regions in blue are associated with a lower probability for AD given high intensities in the voxels.")

    st.info("If you have any questions or want to dive deeper into the mathematics, feel free to explore our linked publication in the sidebar!")
import numpy as np
import pickle as pkl
import streamlit as st
from streamlit_option_menu import option_menu
import warnings

warnings.filterwarnings("ignore")

# Loading predictive_models...
heart_model = open('heart_model.pkl', 'rb')
heart_classifier = pkl.load(heart_model)
breast_cancer_model = open('breast_cancer.pkl', 'rb')
breast_cancer_classifier = pkl.load(breast_cancer_model)
diabetes_model = open('diabetes.pkl', 'rb')
diabetes_classifier = pkl.load(diabetes_model)


class DiseaseClassification:

    def __init__(self):
        with st.sidebar:
            self.user_menu = option_menu(
                menu_title=None,
                options=['Home', 'Heart Attack', 'Breast Cancer', 'Diabetes', 'About'],
                icons=['house-fill', 'activity', 'snow3', 'droplet-fill', 'info-circle'],
                menu_icon='cast',
                default_index=0,
                orientation='vertical'
            )

        if self.user_menu == 'Home':
            self.home_page()
        if self.user_menu == 'Heart Attack':
            self.get_heart_features()
        if self.user_menu == 'Breast Cancer':
            self.get_breast_cancer_features()
        if self.user_menu == 'Diabetes':
            self.get_diabetes_features()
        if self.user_menu == 'About':
            self.about_page()

    def home_page(self):
        st.markdown("<h1 style='text-align: center'> Disease Classification System"
                    "</h1><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify'> "
                    "In the era of Artificial Intelligence and Machine Learning, "
                    "there are a lot of fields in which AI/ML is used today, "
                    "and the healthcare sector is one of the most popular among all. "
                    "In order to contribute to our nation in the healthcare sector, "
                    "we developed a mini project called \"Disease Classification System\".\n\n"
                    "<b> The main goal of this platform is to classify the below diseases in humans:-</b>\n"
                    "1. Heart attack\n2. Breast Cancer\n3. Diabetes\n\n"
                    "</p><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify'> <b>1. Heart Attack: </b>"
                    "A heart attack, also called a myocardial infarction, is a serious medical condition that occurs "
                    "when the blood flow to a part of the heart is blocked, causing damage to the heart muscle. "
                    "The most common cause of a heart attack is a blockage in one or more of the coronary arteries, "
                    "which supply blood to the heart.\n\n Symptoms of a heart attack can include chest pain or discomfort, "
                    "shortness of breath, nausea, etc. If you suspect that you or someone you know is having a heart "
                    "attack, it is important to seek medical attention immediately. There are several risk factors for "
                    "heart attack, including high blood pressure, high cholesterol, smoking, obesity, diabetes, and a "
                    "family history of heart disease. Some lifestyle changes that may help to reduce the risk of heart "
                    "attack include eating a healthy diet, exercising regularly, not smoking, and managing stress. "
                    "Treatment for a heart attack may include medications to break up or dissolve the blockage in the "
                    "coronary artery, procedures to open the artery and restore blood flow, or surgery to repair or "
                    "bypass the damaged artery. It is important to work with a healthcare team to determine the best "
                    "course of treatment for your individual situation.\n\n"
                    "</p><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify'> <b>2. Breast Cancer: </b> "
                    "Breast cancer is a type of cancer that begins in the cells of the breast. "
                    "It can occur in both men and women, but it is much more common in women."
                    "There are several types of breast cancer, including ductal carcinoma, lobular carcinoma, "
                    "and invasive breast cancer. Ductal carcinoma begins in the cells that line the milk ducts in "
                    "the breast, and lobular carcinoma begins in the cells that line the lobules (milk-producing glands)"
                    " in the breast. Invasive breast cancer is a type of breast cancer that has spread from the ducts or"
                    " lobules to the surrounding tissue in the breast. Symptoms of breast cancer may include a lump or "
                    "thickening in the breast or underarm, changes in the size or shape of the breast, dimpling of the "
                    "skin on the breast, and nipple discharge or a change in the way the nipple looks. Risk factors for "
                    "breast cancer include being female, increasing age, a personal or family history of breast cancer,"
                    " certain inherited gene mutations (such as BRCA1 and BRCA2), and certain lifestyle factors (such "
                    "as alcohol consumption and lack of physical activity). Treatment for breast cancer may include "
                    "surgery (such as a lumpectomy or mastectomy), radiation therapy, chemotherapy, hormone therapy, or "
                    "targeted therapy. The specific treatment plan will depend on the type and stage of the cancer, as "
                    "well as the individual's age, overall health, and personal preferences. It is important to work "
                    "with a healthcare team to determine the best course of treatment. Breast cancer is a type of "
                    "cancer that begins in the cells of the breast. It can occur in both men and women, but it is much "
                    "more common in women. There are several types of breast cancer, including ductal carcinoma, "
                    "lobular carcinoma, and invasive breast cancer. Ductal carcinoma begins in the cells that line the"
                    " milk ducts in the breast, and lobular carcinoma begins in the cells that line the lobules "
                    "(milk-producing glands) in the breast. Invasive breast cancer is a type of breast cancer that has"
                    " spread from the ducts or lobules to the surrounding tissue in the breast.\n\n Symptoms of breast "
                    "cancer may include a lump or thickening in the breast or underarm, changes in the size or shape of"
                    " the breast, dimpling of the skin on the breast, and nipple discharge or a change in the way the"
                    " nipple looks. Risk factors for breast cancer include being female, increasing age, a personal or"
                    " family history of breast cancer, certain inherited gene mutations (such as BRCA1 and BRCA2), and"
                    " certain lifestyle factors (such as alcohol consumption and lack of physical activity). "
                    "Treatment for breast cancer may include surgery (such as a lumpectomy or mastectomy), radiation"
                    " therapy, chemotherapy, hormone therapy, or targeted therapy. The specific treatment plan will "
                    "depend on the type and stage of the cancer, as well as the individual's age, overall health, and"
                    " personal preferences. It is important to work with a healthcare team to determine the best course"
                    " of treatment.\n\n"
                    "</p><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify'> <b>3. Diabetes: </b> "
                    "Diabetes is a chronic medical condition in which the body is unable to properly regulate its blood "
                    "sugar (glucose) levels. There are two main types of diabetes: type 1 and type 2.\n\n"
                    "Type 1 diabetes is an autoimmune disorder in which the body's immune system attacks and destroys "
                    "the cells in the pancreas that produce insulin, a hormone that regulates blood sugar. As a result,"
                    " people with type 1 diabetes must take insulin injections or use an insulin pump to manage their"
                    " blood sugar levels.\n\n"
                    "Type 2 diabetes is a metabolic disorder in which the body does not properly use the insulin it "
                    "produces or does not produce enough insulin. It is the most common form of diabetes and is often"
                    " associated with being overweight or obese, physically inactive, and having a family history of"
                    " the disease. People with type 2 diabetes may be able to manage their condition with lifestyle"
                    " changes (such as diet and exercise) and oral medications, but some may also need insulin"
                    " injections.\n\n Symptoms of diabetes can include increased thirst, frequent urination, fatigue,"
                    " blurred vision, and slow healing of cuts and bruises. If left untreated, diabetes can lead to"
                    " serious complications, such as heart disease, nerve damage, kidney disease, blindness, and"
                    " amputation. There is no cure for diabetes, but it can be managed through a combination of "
                    "lifestyle changes (such as diet and exercise) and medical treatment (such as insulin injections "
                    "or oral medications). It is important for people with diabetes to work with a healthcare team to "
                    "regularly monitor their blood sugar levels and manage their condition.\n\n</p><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify'> <h5>Prerequisites: </h5>"
                    " <b>To predict the above three diseases in a human, one must have the required data. "
                    "The data can be accessed from various lab test reports.</b>"
                    "</p>", unsafe_allow_html=True)

    def get_diabetes_features(self):
        st.markdown("<h1 style='text-align: center'> Diabetes Prediction"
                    "</h1>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input('Age', 20, 100)

        with c2:
            bmi = st.number_input('BMI', step=1., format='%.1f')

        with c3:
            glucose = st.number_input('Glucose Level')

        c4, c5, c6 = st.columns(3)
        with c4:
            pregnancies = st.number_input('Number of Pregnancies')

        with c5:
            diabetes_predegree_function = st.number_input('Diabetes Pre-degree', step=1., format='%.3f')

        if st.button('Predict'):
            self.diabetes_prediction(age, bmi, glucose, pregnancies, diabetes_predegree_function)

    def diabetes_prediction(self, age, bmi, glucose, pregnancies, diabetes_predegree_function):
        diabetes_data = (age, bmi, glucose, pregnancies, diabetes_predegree_function)
        features = np.asarray(diabetes_data)
        features = features.reshape(1, -1)
        diabetes_prediction = diabetes_classifier.predict(features)
        if diabetes_prediction[0] == 0:
            st.success('You don\'t have any diabetes issues.')
        elif diabetes_prediction[0] == 1:
            st.success('You are having Diabetes issue. Need to consult with doctor!')

    def get_heart_features(self):
        st.markdown("<h1 style='text-align: center'> Heart Attack Prediction"
                    "</h1>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input('Select Your Age', 20, 80)

        with c2:
            sex = st.selectbox('Gender', ('MALE', 'FEMALE'))

        c3, c4 = st.columns(2)
        with c3:
            exercise_angina = st.selectbox('Exercise Induced Angina',
                                           ('YES', 'NO'))

        with c4:
            fasting_bs = st.selectbox('Fasting Blood Sugar > 120(mg/dl)',
                                      ('True', 'Otherwise'))

        c5, c6 = st.columns(2)
        with c5:
            resting_bp = st.number_input('Resting Blood Pressure(in mmHg)', 0, 200)

        with c6:
            old_peak = st.number_input('Numeric value of ST measured in depression',
                                       -5.0, 15.0, step=0.1)

        if sex == 'MALE':
            sex = 1
        elif sex == 'FEMALE':
            sex = 0

        if exercise_angina == 'YES':
            exercise_angina = 1
        elif exercise_angina == 'NO':
            exercise_angina = 0

        if fasting_bs == 'True':
            fasting_bs = 1
        elif fasting_bs == 'Otherwise':
            fasting_bs = 0

        if st.button('Predict'):
            self.heart_attack_prediction(age, resting_bp, fasting_bs, old_peak, sex, exercise_angina)

    def heart_attack_prediction(self, age, resting_bp, fasting_bs, old_peak, sex, exercise_angina):
        heart_data = (age, resting_bp, fasting_bs, old_peak, sex, exercise_angina)
        features = np.asarray(heart_data)
        features = features.reshape(1, -1)
        heart_attack_prediction = heart_classifier.predict(features)
        if heart_attack_prediction[0] == 0:
            st.success('You will safe from Heart attack!')
        elif heart_attack_prediction[0] == 1:
            st.success('You will face Heart Attack!')

    def get_breast_cancer_features(self):
        st.markdown("<h1 style='text-align: center'> Breast Cancer Prediction"
                    "</h1>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            radius_mean = st.number_input('Radius Mean', step=1., format='%.5f')

        with c2:
            smoothness_worst = st.number_input('Smoothness Worst', step=1., format='%.5f')

        with c3:
            perimeter_mean = st.number_input('Perimeter Mean', step=1., format='%.5f')

        c4, c5, c6 = st.columns(3)
        with c4:
            area_mean = st.number_input('Area Mean', step=1., format='%.5f')

        with c5:
            concavity_mean = st.number_input('Concavity Mean', step=1., format='%.5f')

        with c6:
            concave_points_mean = st.number_input('Concave Points Mean', step=1., format='%.5f')

        c7, c8, c9 = st.columns(3)
        with c7:
            compactness_worst = st.number_input('Compactness Worst', step=1., format='%.5f')

        with c8:
            symmetry_se = st.number_input('Symmetry_se', step=1., format='%.5f')

        with c9:
            radius_worst = st.number_input('Radius Worst', step=1., format='%.5f')

        c10, c11, c12 = st.columns(3)
        with c10:
            perimeter_worst = st.number_input('Perimeter Worst', step=1., format='%.5f')

        with c11:
            area_worst = st.number_input('Area Worst', step=1., format='%.5f')

        with c12:
            concave_points_worst = st.number_input('Concave Points Worst', step=1., format='%.5f')

        if st.button('Predict'):
            return self.breast_cancer_prediction(radius_mean, smoothness_worst, perimeter_mean, area_mean,
                                                 concavity_mean, concave_points_mean, compactness_worst, symmetry_se,
                                                 radius_worst, perimeter_worst, area_worst, concave_points_worst)

    def breast_cancer_prediction(self, radius_mean, smoothness_worst, perimeter_mean, area_mean, concavity_mean,
                                 concave_points_mean, compactness_worst, symmetry_se, radius_worst, perimeter_worst,
                                 area_worst, concave_points_worst):
        breast_cancer_data = (radius_mean, smoothness_worst, perimeter_mean, area_mean, concavity_mean,
                              concave_points_mean, compactness_worst, symmetry_se, radius_worst, perimeter_worst,
                              area_worst, concave_points_worst)
        breast_cancer_features = np.asarray(breast_cancer_data)
        breast_cancer_features = breast_cancer_features.reshape(1, -1)
        predict_breast_cancer = breast_cancer_classifier.predict(breast_cancer_features)
        if predict_breast_cancer[0] == 0:
            st.success('Cancer Type: Malignant')
        elif predict_breast_cancer[0] == 1:
            st.success('Cancer Type: Benign')

    def about_page(self):
        st.markdown("<h1 style='text-align: center'> About"
                    "</h1><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:justify'>This platform uses various techniques to make predictions about"
                    " different diseases. Mainly it uses Machine Learning algorithms to predict results. Together, "
                    "it uses some of the validation testing methods of machine learning to give better results. If you "
                    "feel any suggestions or corrections then feel free to contact me at the below-given contact "
                    "information.</p> <hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center'>Contact Information</h3><hr>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left'> <b>1. Email address:</b> 'cse19_4919136@geeta.edu.in'\n\n"
                    "<b>2.Contact Number:</b> +91 87809 48614 </p>", unsafe_allow_html=True)


obj = DiseaseClassification()

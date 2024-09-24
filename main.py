import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import webbrowser

# Sidebar
st.sidebar.title("FarmCure")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"  # Ensure this image exists in your project directory
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this [GitHub repo](https://github.com/your-repo-link).

    This dataset consists of about 87K RGB images of healthy and diseased crop leaves which are categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.

    A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    1. **Train** (70,295 images)
    2. **Test** (33 images)
    3. **Validation** (17,572 images)
    """)

elif app_mode == "Disease Recognition":
    # Dictionary mapping diseases to solutions
    dict1 = {
        "Apple___Apple_scab": ["Apple", "Apple Scab",
                               "https://www.independenttree.com/apple-scab-identification-prevention-treatment-2/",
                               "https://www.youtube.com/results?search_query=apple+scab+solution"],
        "Apple___Black_rot": ["Apple", "Apple Black Rot",
                              "https://extension.wvu.edu/lawn-gardening-pests/plant-disease/tree-fruit-disease/black-rot-disease-in-apples",
                              "https://www.youtube.com/results?search_query=apple+black+rot+solution"],
        "Apple___Cedar_apple_rust": ["Apple", "Apple Cedar Rust",
                                     "https://www.planetnatural.com/pest-problem-solver/plant-disease/cedar-apple-rust/",
                                     "https://www.youtube.com/results?search_query=apple+cedar+rust+solution"],
        "Apple___healthy": ["Apple", "Healthy",
                            "https://portal.ct.gov/caes/fact-sheets/plant-pathology/disease-control-for-home-apple-orchards#:~:text=Sanitation%20through%20the%20removal%20of,refer%20to%20Spray%20Guide%20below).",
                            "https://www.youtube.com/results?search_query=Prevent+apple+Diseases"],
        "Blueberry___healthy": ["Blueberry", "Healthy",
                                "https://portal.ct.gov/caes/fact-sheets/plant-pathology/disease-control-for-the-home-blueberry-planting#:~:text=Fortunately%2C%20mummy%20berry%20is%20not,the%20vicinity%20of%20the%20planting.",
                                "https://www.youtube.com/results?search_query=Prevent+Bluberry+Diseases"],
        "Cherry_(including_sour)___Powdery_mildew": ["Cherry (including_sour)", "Cherry Powdery Mildew",
                                                     "https://treefruit.wsu.edu/crop-protection/disease-management/cherry-powdery-mildew/",
                                                     "https://www.youtube.com/results?search_query=Cherry+Powdery+Mildew+solution"],
        "Cherry_(including_sour)___healthy": ["Cherry (including_sour)", "Healthy",
                                              "https://www.picturethisai.com/care/Prunus_cerasus.html#:~:text=Solutions%3A%20In%20the%20case%20of,leaves%20in%20dry%2C%20cool%20weather.",
                                              "https://www.youtube.com/results?search_query=Prevent+sour+cherry+Diseases"],
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ["Corn (maize)",
                                                               "Corn Cercospora Leaf Spot / Gray Leaf Spot",
                                                               "https://www.pioneer.com/us/agronomy/gray_leaf_spot_cropfocus.html#:~:text=Fungicides,an%20economical%20solution%20to%20GLS.",
                                                               "https://www.youtube.com/results?search_query=Corn+Cercospora+Leaf+Spot+solution"],
        "Corn_(maize)___Common_rust_": ["Corn (maize)", "Corn Common Rust",
                                        "https://www.pioneer.com/us/agronomy/common_rust_corn_cropfocus.html",
                                        "https://www.youtube.com/results?search_query=Corn+Common+Rust+solution"],
        "Corn_(maize)___Northern_Leaf_Blight": ["Corn (maize)", "Corn Northern Leaf Blight",
                                                "https://www.pioneer.com/us/agronomy/Managing-Northern-Corn-Leaf-Blight.html",
                                                "https://www.youtube.com/results?search_query=Corn+Northern+Leaf+Blight+solution"],
        "Corn_(maize)___healthy": ["Corn (maize)", "Healthy",
                                   "https://www.aflatoxinpartnership.org/wp-content/uploads/2021/05/MAIZE-DISEASES-PDF.pdf",
                                   "https://www.youtube.com/results?search_query=Prevent+Corn+Maize+Diseases"],
        "Grape___Black_rot": ["Grape", "Grape Black Rot",
                              "https://extension.psu.edu/black-rot-on-grapes-in-home-gardens",
                              "https://www.youtube.com/results?search_query=Grape+Black+Rot+solution"],
        "Grape___Esca_(Black_Measles)": ["Grape", "Grape Esca Black Measles",
                                         "https://grapes.extension.org/grapevine-measles/",
                                         "https://www.youtube.com/results?search_query=Grape+Esca+Black+Measles+solution"],
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ["Grape", "Grape Leaf Blight Isariopsis Leaf Spot",
                                                       "https://www.sciencedirect.com/science/article/abs/pii/S0261219414001598",
                                                       "https://www.youtube.com/results?search_query=Grape+Leaf+Blight+Isariopsis+Leaf+Spot+solution"],
        "Grape___healthy": ["Grape", "Healthy",
                            "https://portal.ct.gov/caes/fact-sheets/plant-pathology/disease-control-for-home-grape-plantings#:~:text=Grape%20diseases%20can%20be%20effectively,high%20percentage%20of%20quality%20berries.",
                            "https://www.youtube.com/results?search_query=Prevent+Grape+Diseases"],
        "Orange___Haunglongbing_(Citrus_greening)": ["Orange", "Orange Huaunglongbing Citrus Greening",
                                                     "https://www.aphis.usda.gov/plant-pests-diseases/citrus-diseases/citrus-greening-and-asian-citrus-psyllid#:~:text=Citrus%20greening%2C%20also%20called%20Huanglongbing,There%20is%20no%20cure.",
                                                     "https://www.youtube.com/results?search_query=Orange+Huaunglongbing+Citrus+Greening+solution"],
        "Peach___Bacterial_spot": ["Peach", "Peach Bacterial Spot",
                                   "https://www.aces.edu/blog/topics/crop-production/bacterial-spot-treatment-in-peaches/#:~:text=Management%20of%20this%20disease%20often,could%20benefit%20from%20spray%20applications.",
                                   "https://www.youtube.com/results?search_query=Peach+Bacterial+Spot+solution"],
        "Peach___healthy": ["Peach", "Healthy",
                            "https://eudyan.hp.gov.in/UploadedFiles/sprayshedule/new-spray-schedules-for-stone-fruits.pdf",
                            "https://www.youtube.com/results?search_query=Prevent+Peach+Diseases"],
        "Pepper,_bell___Bacterial_spot": ["Pepper Bell", "Pepper Bell Bacterial Spot",
                                          "https://portal.ct.gov/-/media/caes/documents/publications/fact_sheets/plant_pathology_and_ecology/bacterialspotofpepper032912pdf.pdf",
                                          "https://www.youtube.com/results?search_query=Pepper+Bell+Bacterial+Spot+solution"],
        "Pepper,_bell___healthy": ["Pepper Bell", "Healthy", "https://www.fao.org/4/y5259e/y5259e0b.htm",
                                   "https://www.youtube.com/results?search_query=Prevent+Pepper+Bell+Diseases"],
        "Potato___Early_blight": ["Potato", "Potato Early Blight",
                                  "https://ipm.ucanr.edu/agriculture/potato/early-blight/#gsc.tab=0",
                                  "https://www.youtube.com/results?search_query=Potato+Early+Blight+solution"],
        "Potato___Late_blight": ["Potato", "Potato Late Blight",
                                 "https://www.lfl.bayern.de/ips/blattfruechte/034444/index.php#:~:text=If%20there%20is%20visible%20late,one%20of%20these%20two%20fungicides.",
                                 "https://www.youtube.com/results?search_query=Potato+Late+Blight+solution"],
        "Potato___healthy": ["Potato", "Healthy", "https://nhb.gov.in/pdf/vegetable/potato/pot002.pdf",
                             "https://www.youtube.com/results?search_query=Prevent+potato++Diseases"],
        "Raspberry___healthy": ["Raspberry", "Healthy", "https://www.rhs.org.uk/fruit/raspberries/grow-your-own",
                                "https://www.youtube.com/results?search_query=Prevent+Raspberry+Diseases"],
        "Soybean___healthy": ["Soybean", "Healthy",
                              "https://www.corteva.us/Resources/crop-protection/disease-mgmt/prevent-control-soybean-diseases.html#:~:text=Prevention%20and%20control&text=%E2%80%9CFocus%20on%20seed%20treatment%20with,are%20either%20metalaxyl%20or%20mefenoxam.",
                              "https://www.youtube.com/results?search_query=Prevent+Soyabean+Diseases"],
        "Squash___Powdery_mildew": ["Squash", "Squash Powdery Mildew",
                                    "https://www.growveg.com/plant-diseases/us-and-canada/squash-powdery-mildew/#:~:text=Thin%20plants%20to%20proper%20spacing,kind)%20to%20four%20parts%20water.",
                                    "https://www.youtube.com/results?search_query=Squash+Powdery+Mildew+solution"],
        "Strawberry___Leaf_scorch": ["Strawberry", "Strawberry Leaf Scorch",
                                     "https://content.ces.ncsu.edu/leaf-scorch-of-strawberry",
                                     "https://www.youtube.com/results?search_query=Strawberry+Leaf+Scorch+solution"],
        "Strawberry___healthy": ["Strawberry", "Healthy",
                                 "https://www.koppert.com/plant-diseases/leaf-spot-of-strawberry/",
                                 "https://www.youtube.com/results?search_query=Prevent+Strawberry+Diseases"],
        "Tomato___Bacterial_spot": ["Tomato", "Tomato Bacterial Spot",
                                    "https://portal.ct.gov/-/media/caes/documents/publications/fact_sheets/plant_pathology_and_ecology/2018/bacterialspotoftomatopdf.pdf",
                                    "https://www.youtube.com/results?search_query=Tomato+Bacterial+Spot+solution"],
        "Tomato___Early_blight": ["Tomato", "Tomato Early Blight",
                                  "https://extension.umn.edu/disease-management/early-blight-tomato-and-potato",
                                  "https://www.youtube.com/results?search_query=Tomato+Early+Blight+solution"],
        "Tomato___Late_blight": ["Tomato", "Tomato Late Blight",
                                 "https://vegpath.plantpath.wisc.edu/diseases/tomato-late-blight/#:~:text=Strategies%20for%20managing%20late%20blight,effective%20fungicides%20to%20avoid%20infection.",
                                 "https://www.youtube.com/results?search_query=Tomato+Late+Blight+solution"],
        "Tomato___Leaf_Mold": ["Tomato", "Tomato Leaf Mold",
                               "https://www.gardeningknowhow.com/edible/vegetables/tomato/managing-tomato-leaf-mold.htm#:~:text=Tomato%20Leaf%20Mold%20Treatment,-The%20pathogen%20P&text=High%20relative%20humidity%20(greater%20that,temps%20higher%20than%20outside%20temperatures.",
                               "https://www.youtube.com/results?search_query=Tomato+Leaf+Mold+solution"],
        "Tomato___Septoria_leaf_spot": ["Tomato", "Tomato Septoria Leaf Spot",
                                        "https://content.ces.ncsu.edu/septoria-leaf-spot-of-tomato",
                                        "https://www.youtube.com/results?search_query=Tomato+Septoria+Leaf+Spot+solution"],
        "Tomato___Spider_mites Two-spotted_spider_mite": ["Tomato", "Tomato Two-spotted Spider Mite",
                                                          "https://agrilifeorganic.org/2024/05/20/spider-mites-on-tomato/",
                                                          "https://www.youtube.com/results?search_query=Tomato+Two-spotted+Spider+Mite+solution"],
        "Tomato___Target_Spot": ["Tomato", "Tomato Target Spot",
                                 "https://apps.lucidcentral.org/pppw_v10/text/web_full/entities/tomato_target_spot_163.htm#:~:text=Warm%20wet%20conditions%20favour%20the,4%20weeks%20before%20last%20harvest.",
                                 "https://www.youtube.com/results?search_query=Tomato+Target+Spot+solution"],
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["Tomato", "Tomato Yellow Leaf Curl Virus",
                                                   "https://plantix.net/en/library/plant-diseases/200036/tomato-yellow-leaf-curl-virus/",
                                                   "https://www.youtube.com/results?search_query=Tomato+Yellow+Leaf+Curl+Virus+solution"],
        "Tomato___Tomato_mosaic_virus": ["Tomato", "Tomato Mosaic Virus",
                                         "https://www.missouribotanicalgarden.org/gardens-gardening/your-garden/help-for-the-home-gardener/advice-tips-resources/insects-pests-and-problems/diseases/viruses/tobacco-mosaic-virus",
                                         "https://www.youtube.com/results?search_query=Tomato+Mosaic+Virus+solution"],
        "Tomato___healthy": ["Tomato", "Healthy", "https://abundantminigardens.com/prevent-tomato-diseases/",
                             "https://www.youtube.com/results?search_query=Prevent+Tomato+Diseases"]
    }


    # Function to check if the image contains a leaf based on green pixels percentage
    def check_if_leaf(image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_green = cv2.inRange(hsv, (36, 0, 0), (86, 255, 255))
        green_pixels = np.sum(mask_green > 0)
        total_pixels = image.shape[0] * image.shape[1]
        green_percentage = green_pixels / total_pixels
        return green_percentage * 100  # Return percentage of green pixels


    # Function to sharpen the image
    def sharpen_image(image):
        image_np = np.array(image)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened_image = cv2.filter2D(image_np, -1, kernel)
        return sharpened_image


    # Function to load and predict using the model
    def model_prediction(model, test_image):
        image = Image.open(test_image).resize((224, 224))  # Resize to the correct dimensions
        input_arr = np.array(image) / 255.0  # Normalize image
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
        input_arr = input_arr.reshape((1, 224, 224, 3))  # Ensure correct shape
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element


    # Function to map disease to dictionary and get solution links
    def get_solution_links(class_name):
        if class_name in dict1:
            disease_info = dict1[class_name]
            return disease_info[2], disease_info[3]  # Return More Info URL and Video Tutorial URL
        else:
            return None, None


    # Load the model
    try:
        model = tf.keras.models.load_model("/Users/sankalp/Downloads/plant_disease_prediction_model.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # Streamlit App
    st.title("DISEASE RECOGNITION")

    # Subheader for the input section with reduced font size using HTML
    st.markdown("<h3 style='font-size:20px;'>Upload an Image of a Leaf</h3>", unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Convert the uploaded file to a numpy array
            image = np.array(Image.open(uploaded_file))

            # Convert to BGR for OpenCV processing
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            st.markdown("<h3 style='font-size:20px;'>Processing Results</h3>", unsafe_allow_html=True)

            # Check if the image contains a leaf
            green_percentage = check_if_leaf(image_bgr)

            if green_percentage > 30:  # Threshold to consider it a leaf
                st.write(
                    f"**Healthy Percentage:** {green_percentage:.2f}% | **Diseased Percentage:** {100 - green_percentage:.2f}%")

                # Convert the image to its negative
                negative_img = cv2.bitwise_not(image_bgr)

                # Sharpen the original image
                sharpened_img = sharpen_image(image)

                # Convert back to RGB for displaying in Streamlit
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                negative_rgb = cv2.cvtColor(negative_img, cv2.COLOR_BGR2RGB)
                sharpened_rgb = cv2.cvtColor(sharpened_img, cv2.COLOR_BGR2RGB)

                # Display the original, sharpened, and negative images horizontally
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image_rgb, caption="Original Image", use_column_width=True)
                with col2:
                    st.image(sharpened_rgb, caption="Sharpened Image", use_column_width=True)
                with col3:
                    st.image(negative_rgb, caption="Negative Image", use_column_width=True)

                # Predict Disease Button
                if st.button("Predict Disease"):
                    with st.spinner("Model is predicting..."):
                        result_index = model_prediction(model, uploaded_file)
                        class_names = [
                            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Common_rust_',
                            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Northern_Leaf_Blight',
                            'Corn_(maize)___healthy',
                            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
                            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                            'Tomato___Tomato_mosaic_virus',
                            'Tomato___healthy'
                        ]
                        try:
                            disease_name = class_names[result_index]
                            st.success(f"**Model predicts:** {disease_name}")

                            # Store prediction in session state
                            st.session_state['prediction'] = disease_name
                        except IndexError:
                            st.error("Prediction index is out of range. Please check your class names.")
                        except Exception as e:
                            st.error(f"An error occurred during prediction: {e}")

                # Display "Find More" button only if a prediction has been made
                if 'prediction' in st.session_state:
                    if st.button("Find More"):
                        class_name = st.session_state['prediction']
                        # Get the corresponding class key from dict1
                        # Assuming class_name is in the format "Plant___Disease"
                        if class_name in dict1:
                            more_info_url, video_url = dict1[class_name][2], dict1[class_name][3]
                            st.markdown(f"**More Information:** [Click here]({more_info_url})")
                            st.markdown(f"**Video Tutorials:** [Click here]({video_url})")
                        else:
                            st.write("No additional information found for the predicted disease.")

            else:
                st.error("The uploaded image does not appear to be a leaf. Please upload a proper leaf image.")
        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")

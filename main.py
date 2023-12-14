import streamlit as st
import pickle
import nltk# Path to your NLTK data directory (AppData location)
import os
nltk_data_dir = os.path.expanduser('~\\AppData\\Roaming\\nltk_data')

# Add the NLTK data directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

# Check if stopwords and punkt are present in the nltk_data_dir
stopwords_present = os.path.exists(os.path.join(nltk_data_dir, 'corpora/stopwords'))
punkt_present = os.path.exists(os.path.join(nltk_data_dir, 'tokenizers/punkt'))

if not stopwords_present:
    # If stopwords not present, download them
    nltk.download('stopwords', download_dir=nltk_data_dir)

if not punkt_present:
    # If punkt not present, download it
    nltk.download('punkt', download_dir=nltk_data_dir)
    
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import os


# Set the NLTK data path
nltk.data.path.append(nltk_data_dir)
# Define your preprocessing function (this is just a placeholder, replace with your actual preprocessing)
ps = PorterStemmer()

def transform_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = "".join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Stem the words
    stemmed_words = [ps.stem(word) for word in words]

    # Join the words back into a string
    text = " ".join(stemmed_words)
    
    return text
# Assuming you've already done preprocessing similar to spam classifier
# and your sentiment analysis model is trained with the same preprocessing
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

#for importing back ground dont confuse
import base64
import streamlit as st

# Function to convert file to base64
def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your image
image_path = r"sentimental.jpg"


# Convert the image to base64
image_base64 = get_image_as_base64(image_path)

# Function to add background from local
def add_bg_from_base64(base64_string):
    # The corrected CSS string with the base64 image
    css_string = f'''
    .stApp {{
        background-image: url("data:image/jpg;base64,{base64_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    '''
    # Using st.markdown to inject the CSS string with the base64 image
    st.markdown(f'<style>{css_string}</style>', unsafe_allow_html=True)

    #     f"""
    #     <style>
        # .stApp {{
        #     background-image: url("data:image/jpg;base64,{base64_string}");
        #     background-size: cover;
        #     background-repeat: no-repeat;
        #     background-attachment: fixed;
    #     }}

    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

# Call the function to add the background
add_bg_from_base64(image_base64)

with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.title("Sentiment Analysis Classifier")

input_text = st.text_area("Enter your text sentiment analysis") 

if st.button('Analyze Sentiment'):
    # 1. Preprocess the text (make sure this matches the preprocessing done during model training)
    transformed_text = transform_text(input_text)  # Assuming transform_text is your preprocessing function
    
    # 2. Vectorize the processed text
    vector_input = tfidf.transform([transformed_text])
    
    # 3. Predict sentiment
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header('Positive Sentiment 👍')
    else:
        st.header("Negative Sentiment 👎")


   

def main():
    # Your Streamlit app code here
    st.write('MADE BY SRMD')
    st.write('This APP helps you to analyse ur text')
    st.write('Thank me later')
   


if __name__ == '__main__':
    main()



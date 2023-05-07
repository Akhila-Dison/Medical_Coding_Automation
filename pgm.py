# & "mc\Scripts\Activate.ps1"
import pickle
from logging import debug
import PyPDF2
import pandas as pd
import nltk
from nltk.corpus import stopwords
import scispacy
import spacy
nltk.download('stopwords')
import re
nlp = spacy.load("en_core_sci_scibert")
nltk.download('punkt')
from spacy.pipeline.textcat import Config, single_label_cnn_config
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import train_test_split
import joblib
le=joblib.load('labelEncoder.joblib')
nlpc = spacy.load("models")
with open('dfs.pkl', 'rb') as fp:
    dfs = pickle.load(fp)
# Reading pdf file
def pdf_read(file_name):
    # Extracting the medical report pdf content.
    reader = PyPDF2.PdfReader(file_name)
    pdf_length = len(reader.pages)
    texts = ""
    for pdf_pages in range(pdf_length):
        page_content = reader.pages[pdf_pages].extract_text()
        texts = texts + page_content
    return texts
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
def remove_stopwords(text):
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])

def remove_punctuations(text_s):
    return re.sub('[^a-zA-Z]', '', text_s)


# Reading dataset
data = pd.read_csv('ICD_10.csv')
cols = data.select_dtypes(object).columns.difference(['sub', 'status'])
data[cols] = data[cols].apply(lambda x: x.str.lower())
medical_list = data['M_Term'].str.lower().to_list()

from flask import Flask,render_template,request

import time

app = Flask(__name__,template_folder='.',static_folder='.')


@app.route('/',methods=['GET','POST'])
def index():
    result =''
    if request.method == 'POST':
            files = request.files['file']
            filename = files.filename
            filepath = './'+filename
            files.save(filepath)
            print('file saved')
            time.sleep(2)
            text = pdf_read(filepath)
            # Convert upper cases to lower
            text = text.lower()

            # Remove Stop words (commonly used words in a language)
            stop_words = stopwords.words('english')
            text = remove_stopwords(text)
            docx = nlp(text)
            ents = docx.ents
            ents = [ent.text.strip() for ent in ents]
            list_one = set(ents)
            list_two = set(medical_list)
            mt = list_one & list_two
            
                
            if len(mt)==0:
                for items in medical_list:
                    # item = re.search(re.escape(repr(items)), text)
                    items = items.lower()
                    item = re.search(items, text)
                    if item is not None:
                        mt = [item.group()]
            if len(mt)==0:
                result = "failed to process the pdf"
            else:
                mt = list(mt)[0]
                doc=nlpc(mt)
                res= doc.cats  
                res = sorted(res.items(), key=lambda x: x[1])
                out = int(res[-1][0].split("_")[-1])
                mcc = le.inverse_transform([out])[0]
                df = dfs[mcc]
                df1 = df.loc[df['M_Term'].str.strip()== mt.strip()]
                mc = df1["M_Code"].to_list()[0]
                mtc = df1["M_Term_Category"].to_list()[0]
                result = f"Medical_Code_Category : {mcc},  Medical_Code :{mc},  Medical_Term_Category:{mtc}, Medcial_Term:{mt}"
    return render_template('index.html',result=result)

app.run(debug=True)


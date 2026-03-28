import streamlit as st
import joblib
import re
import pandas as pd

def mycleaning(doc):
    return re.sub("[^a-zA-Z ]","",doc).lower()

model=joblib.load("sentiment_model.pkl") 

st.set_page_config(layout='wide')
st.markdown("""
    <div style="
        background: linear-gradient(90deg, #ffff00, #2E7D32);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    ">
        <h1 style="
            color: purple;
            font-size: 40px;
            margin: 0;
        ">
            Food Sentiment Analysis
        </h1>
    </div>
""", unsafe_allow_html=True)

# st.sidebar.image("new_rohit.jpg")

st.sidebar.title("About Project")
st.sidebar.write("Prediction of sentiment Neg or pos for a food review")

st.sidebar.title("Contact us,📞")
st.sidebar.write("999999999")

st.sidebar.title("About us 👩‍💻")
st.sidebar.write("we are a group of ai engineers at ducat")

st.write("\n")
st.write('#### Enter Review')
sample=st.text_input("")
if st.button("predict"):
    pred=model.predict([sample])
    prob=model.predict_proba([sample])
    if pred[0]==0:
        st.write("Neg👎")
        st.write(f"Confidence Score:{prob[0][0]:.2f}")
    else:
        st.write("Pos👍")
        st.write(f"Confidence Score:{prob[0][0]:.2f}")
        st.balloons()

st.write("#### Bulk Prediction")
file=st.file_uploader("select file",type=["csv","txt"])
if file:
    df=pd.read_csv(file,names=["Review"])
    placeholder=st.empty()
    placeholder.dataframe(df)
    if st.button("Predict",key="b2"):
        corpus=df.Review
        pred=model.predict(corpus)
        prob=np.max(model.predict_proba(corpus),axis=1)
        df['Sentiment']=pred
        df['Confidence']=prob
        df['Sentiment']=df['Sentiment'].map({0:'Neg👎',1:'Pos👍'})
        placeholder.dataframe(df)
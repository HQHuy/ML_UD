from flask import Flask, render_template, request
from model import train_gaussian_nb, clean_text
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Load dataset and train models once at startup
data = pd.read_csv('/Users/hqhyy/Desktop/University/Learn IT/Practice/ML_UD/lab2/Education.csv')
data['Cleaned_Text'] = data['Text'].apply(clean_text)

# Train Bernoulli and Multinomial Naive Bayes models
vecto_hoa_van_ban_Bernoulli = CountVectorizer(binary=True)
vecto_hoa_van_ban_Multinomial = CountVectorizer()

model_1 = make_pipeline(vecto_hoa_van_ban_Bernoulli, BernoulliNB())
model_2 = make_pipeline(vecto_hoa_van_ban_Multinomial, MultinomialNB())

model_1.fit(data['Cleaned_Text'], data['Label'])
model_2.fit(data['Cleaned_Text'], data['Label'])

@app.route('/', methods=['GET', 'POST'])
def emotions():
    prediction_bernoulli = None
    prediction_multinomial = None
    
    if request.method == 'POST':
        user_input = request.form['user_text']
        cleaned_text = clean_text(user_input)
        
        # Get predictions from both models
        prediction_bernoulli = model_1.predict([cleaned_text])[0]
        prediction_multinomial = model_2.predict([cleaned_text])[0]

    return render_template('index.html', 
                           prediction_bernoulli=prediction_bernoulli, 
                           prediction_multinomial=prediction_multinomial)


drug_data = pd.read_csv('/Users/hqhyy/Desktop/University/Learn IT/Practice/ML_UD/lab2/drug.csv')
drug_model, le_sex, le_bp, le_cholesterol = train_gaussian_nb(drug_data)  # Unpack all three values

@app.route('/drug', methods=['GET', 'POST'])
def drug():
    drug_prediction = None
    
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        bp = request.form['bp']
        cholesterol = request.form['cholesterol']
        na_to_k = float(request.form['na_to_k'])
        
        # Prepare the input data for the model
        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': le_sex.transform([sex]),  # Use the fitted label encoder for 'Sex'
            'BP': le_bp.transform([bp]),      # Use the fitted label encoder for 'BP'
            'Cholesterol': le_cholesterol.transform([cholesterol]),  # Use the fitted label encoder for 'Cholesterol'
            'Na_to_K': [na_to_k]
        })
        
        # Make prediction using GaussianNB
        drug_prediction = drug_model.predict(input_data)[0]
    
    return render_template('drug.html', 
                           drug_prediction=drug_prediction)


if __name__ == '__main__':
    app.run(debug=True)

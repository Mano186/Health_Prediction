from flask import Flask, render_template, request
from flask import send_file
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from statistics import mode
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import os
import seaborn as sns

static_path = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_path):
    os.makedirs(static_path)


app = Flask(__name__)

# Load your data and models
DATA_PATH = "Training.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = 0.2, random_state = 24)

final_svm_model = SVC()
final_nb_model = GaussianNB()
final_knn_model = KNeighborsClassifier(n_neighbors=5)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_knn_model.fit(X, y)

symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}


def predict_disease(symptoms):
    symptoms = [symptom.strip().lower() for symptom in symptoms.split(",")]
    input_data = [0] * len(data_dict["symptom_index"])

    for symptom in symptoms:
        standardized_symptom = " ".join([i.capitalize() for i in symptom.split("_")])
        if standardized_symptom in data_dict["symptom_index"]:
            index = data_dict["symptom_index"][standardized_symptom]
            input_data[index] = 1

    input_data = np.array(input_data).reshape(1, -1)

    knn_prediction = data_dict["predictions_classes"][final_knn_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]

    final_prediction = mode([knn_prediction, nb_prediction, svm_prediction])

    predictions = {
        "knn_model_prediction": knn_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }

    return predictions


@app.route('/')
def index():
    return render_template('index.html')


# ... (previous code)

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        user_input = request.form['symptoms']
        result = predict_disease(user_input)

        # Extract form data for passing to view_report page
        form_data = {
            'title': request.form.get('title', ''),
            'first_name': request.form.get('first_name', ''),
            'last_name': request.form.get('last_name', ''),
            'age': request.form.get('age', ''),
            'gender': request.form.get('gender', ''),
            'symptoms': user_input,
        }

        # Render the result page with the result and a link to the view_report page
        return render_template('result.html', result=result, **form_data)

    # If the request method is not POST, render an error message
    return render_template('result.html', result="Invalid request method.")

@app.route('/download_report')
def download_report():
    # Get the report content and patient's first name from the query parameters
    report_content = request.args.get('report_content', '')
    first_name = request.args.get('first_name', 'Patient')

    # Sanitize the first name to create a safe file name
    safe_first_name = secure_filename(first_name)

    # Customize the file name with the sanitized patient's first name
    file_name = f"{safe_first_name}_medical_report.txt"
    temp_file_path = f'static/{file_name}'

    # Save the report content to a temporary file
    with open(temp_file_path, 'w') as file:
        file.write(report_content)

    # Ensure the file is closed before sending it for download
    file.close()

    # Send the file as an attachment for download
    return send_file(temp_file_path, as_attachment=True, download_name=file_name)



def generate_medical_report(patient_details, predictions):
    title = patient_details.get('title', '')
    full_name = f"{title} {patient_details['first_name']} {patient_details['last_name']}\n"
    
    report_content = f"Full Name: {full_name}"
    report_content += f"Age: {patient_details['age']}\n"
    report_content += f"Gender: {patient_details['gender']}\n"
    
    # Add symptoms to the report
    symptoms = patient_details.get('symptoms', '')
    report_content += f"Symptoms: {symptoms}\n"

    # Add a line break before the "Prediction Details" section
    report_content += "\nPrediction Details:\n"
    
    report_content += f"KNN Model Prediction: {predictions['knn_model_prediction']}\n"
    report_content += f"Naive Bayes Prediction: {predictions['naive_bayes_prediction']}\n"
    report_content += f"SVM Model Prediction: {predictions['svm_model_prediction']}\n"
    report_content += f"Final Prediction: {predictions['final_prediction']}\n"

    # Add more information to the report as needed

    # For debugging, print the report content
    print("Generated Medical Report Content:")
    print(report_content)

    return report_content

# ... (rest of the code)

# ... (rest of the code)

# ... (previous routes and setup)

# ... (previous code)

# ... (previous code)

# ... (previous code)

@app.route('/view_report', methods=['POST'])
def view_report():
    if request.method == 'POST':
        # Use the form data to generate the medical report
        title = request.form['title']
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        gender = request.form['gender']
        symptoms = request.form['symptoms']

        # Perform health prediction or other processing here
        predictions = predict_disease(symptoms)

        report_content = generate_medical_report({
            'title': title,
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'symptoms': symptoms,
        }, predictions)

        # Render the view_report page with the medical report content
        return render_template('view_report.html', report_content=report_content)

    # If the request method is not POST, render an error message
    return render_template('view_report.html', report_content="Invalid request method.")

# ... (rest of the code)


# ... (rest of the code)





    



def generate_confusion_matrix(y_true, y_pred, classifier_name):
    cf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cf_matrix, annot=True)
    plt.title(f"Confusion Matrix for {classifier_name}")
    plt.savefig(f'static/confusion_matrix_{classifier_name.lower()}.png')
    plt.close()

@app.route('/report')
def report():
    # Assuming y_test and preds are defined somewhere in your code
    #KNN predictions
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    preds_knn = nb_model.predict(X_test)
    #KNN predictions
    knn_model = KNeighborsClassifier(n_neighbors=5)  
    knn_model.fit(X_train, y_train)
    preds_nb = knn_model.predict(X_test)
    #SVM predictions
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    preds_svm = svm_model.predict(X_test) 
    #combined model
    final_preds = [mode([i, j, k]) for i, j, k in zip(preds_knn, preds_nb, preds_svm)]
    # Generate confusion matrices
    generate_confusion_matrix(y_test, preds_knn, 'KNN')
    generate_confusion_matrix(y_test, preds_nb, 'NB')
    generate_confusion_matrix(y_test, preds_svm, 'SVM')
    generate_confusion_matrix(y_test, final_preds, 'Combined_Model')

    # Other report generation code
    # ...

    return render_template('report.html')




if __name__ == '__main__':
    app.run(debug=True)

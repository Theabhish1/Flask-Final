import numpy as np
from flask import Flask, request, render_template
import pickle
app=Flask(__name__)
model= pickle.load(open("model.pkl","rb"))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    feature_names = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se",
        "smoothness_se", "compactness_se", "concavity_se",
        "concave points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
        "smoothness_worst", "compactness_worst", "concavity_worst",
        "concave points_worst", "symmetry_worst"
    ]
    float_feature=[float(request.form[feature]) for feature in feature_names]
    features= [np.array(float_feature)]
    prediction=model.predict(features)

    prediction='Cancer' if prediction==1 else 'No Cancer'
    return render_template("index.html", prediction_text="The Predicted Diagnosis is {}".format(prediction))
if __name__== "__main__":
    app.run(debug=True)
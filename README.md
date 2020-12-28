# TenProfiles

This project is Multiclass Classification contains 10 different classes of Twitter profiles between politicians and technologest. The accuracy of the deployed model is 76%. The idea of this project is to shawcase how to use streamlit package as frontend for a deployed model as backend and connect both apps via API

Purpose of this excercise is not to get the perfect result, however, it is to show case the process of building model, exporting it and use it in real world application by deploying it and serve it via API to public. The accuracy of this model is 76.9% and the F1 score is 76.2%.

Try the app via https://streamlit-multiclass-classifie.herokuapp.com/

Steps to enhance model:

- Manually remove common words.
- Use different methods / techniques to deal with unbalanced data
- Use more samples
- Shuffle the dataset before spliting it.
- Use default of increase the max_features parameter in the TF-IDF. Current value is 7000.
- Use more ML algorithms that are suitable for multiclass classification problem. I recommend SVM
- Increase the number of iteration in RandomizedSearchCV
- The last 3 suggestions are computationally expensive, required lots of RAM and it takes time to get results back.


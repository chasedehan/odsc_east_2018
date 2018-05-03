# train_model_odsc.jl
# Show how we load a dataset, train a model, and make predictions

using RDatasets
using DataFrames
using ScikitLearn
using ScikitLearn: fit!, predict
using ScikitLearn.CrossValidation: cross_val_score, cross_val_predict

@sk_import ensemble: GradientBoostingClassifier


# Have access to the rich sklearn functionality
# And you can pass in whatever values you want: folds, metrics, etc.


iris = dataset("datasets", "iris")

X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])



# Then, initialize the model
model = GradientBoostingClassifier()


# How good is the model: a couple methods
# Get the predictions, then score it
y_hat = cross_val_predict(model, X, y)
accuracy = sum(y_hat .== y) /  length(y)

# Or could simply score it
cv_scores = cross_val_score(model, X, y; cv=5)


# Fit the model on all data for production
fit!(model, X, y)


predict(model,X[1,])

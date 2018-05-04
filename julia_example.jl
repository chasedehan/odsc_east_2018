#hello.jl
# Below is the first time I was really working with Julia
# Built models with sklearn and plotted with ggplot
# using DataFrames
using RDatasets
iris = dataset("datasets", "iris")


# ScikitLearn.jl expects arrays, but DataFrames can also be used - see
# the corresponding section of the manual
X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])


using ScikitLearn
using ScikitLearn: fit!, predict
@sk_import linear_model: LogisticRegression

# Then fit the model
model = LogisticRegression(fit_intercept=true)
fit!(model, X, y)

accuracy = sum(predict(model, X) .== y) / length(y)
println("accuracy: $accuracy")



# Then, lets try to plot some stuff
using RCall

R"library(ggplot2);
ggplot($iris, aes(x=SepalLength, y=SepalWidth, color=Species)) + geom_point()"

a = 2
R"multiply <- function(x) x * $a"
@rget multiply

multiply(4)

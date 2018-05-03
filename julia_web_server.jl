# Fashioned into two main parts

###############################################################################
# Train the model
###############################################################################
using RDatasets
using DataFrames
using ScikitLearn
using ScikitLearn: fit!, predict

@sk_import ensemble: GradientBoostingClassifier

iris = dataset("datasets", "iris")
X = convert(Array, iris[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
y = convert(Array, iris[:Species])

model = GradientBoostingClassifier()
fit!(model, X, y)

###############################################################################
# Build the API Call
###############################################################################
using HttpServer
import JSON


function dicttodataframe(dict)
    new_df = DataFrame()
    for (k,v) in dict
        new_df[Symbol(k)] = v
    end
    return new_df
end

http = HttpHandler() do req::Request, res::Response
    if length(req.data) > 0
        requestData = JSON.parse(String(req.data))
        a = dicttodataframe(requestData)
        request_X = convert(Array, a[[:SepalLength, :SepalWidth, :PetalLength, :PetalWidth]])
        println(request_X)
    end

    if req.method == "GET"
        responseData = "Submit as a POST"
    end
    if req.method == "POST"
        y_hat = predict(model, request_X)
        responseData = JSON.json(y_hat)
    end

    Response(responseData)
end


###############################################################################
# Kick off the http server
###############################################################################

server = Server(http)
run(server, 8000)

module MiraculousNN

export fit

using ProgressMeter

include("metrics.jl")
include("forprop.jl")
include("backprop.jl")
include("loss.jl")
include("utils.jl")

using .metrics
using .forprop
using .backprop
using .loss
using .utils


function fit(x, y; epochs = 5, batch_size = 16, learning_rate = 0.01)
    input_size = size(x, 1)
    output_size = size(y, 1)

    nn_params = _init_layers([input_size, 512, 512, output_size])
    nlayers = get_nlayers(nn_params)

    nn_values = Dict()
    grads = Dict()

    num_batches = div(size(x, 2), batch_size)

    for i = 1:epochs
        for b_idx = 1:(num_batches - 1)
            b = b_idx * batch_size
            x_batch = x[:, b:(b + batch_size - 1)]
            y_batch = y[:, b:(b + batch_size - 1)]

            nn_values = _forward(x_batch, nn_params)

            grads = _backward(nn_params, nn_values, x_batch, y_batch)
            _update_weights!(nn_params, grads, learning_rate)
        end

        nn_values = _forward(x, nn_params)
        y_pred = nn_values["A$nlayers"]
        cost = loss.categorical_crossentropy(y, y_pred)
        println("Epoch $i loss: $cost")
    end

    nn_values = _forward(x, nn_params)

    train_accuracy = metrics.accuracy(y, nn_values["A$nlayers"])
    println("Train accuracy: $(train_accuracy)")
end



function _init_layers(layer_sizes)::Dict{String,Matrix{Float64}}
    params = Dict()

    for i = 2:length(layer_sizes)
        params["W$(i - 1)"] = rand(Float64, layer_sizes[i], layer_sizes[i - 1]) * 0.01
        params["B$(i - 1)"] = zeros(Float64, layer_sizes[i], 1)
    end
    return params
end


function _forward(x::T, params::Dict{String,T})::Dict where {T}
    nlayers = get_nlayers(params)
    nn_values = Dict{String,T}()

    last_output = x

    for i = 1:nlayers
        nn_values["Z$i"] = params["W$i"] * last_output .+ params["B$i"]

        activation = i == nlayers ? forprop.softmax : forprop.relu
        nn_values["A$i"] = activation(nn_values["Z$i"])

        last_output = nn_values["A$i"]
    end
    return nn_values
end


function _backward(
    params::Dict{String,T},
    nn_values::Dict{String,T},
    x::T,
    y::Matrix{Int64},
)::Dict{String,T} where {T}
    nlayers = get_nlayers(params)
    nsamples = size(y, 2)
    grads = Dict()

    dA = nn_values["A$nlayers"]
    dZ = backprop.softmax(dA, y)

    for i = nlayers:-1:1
        if i != nlayers
            dA = transpose(params["W$(i + 1)"]) * dZ
            dZ = backprop.relu(dA, nn_values["A$i"])
        end

        last_output = i == 1 ? x : nn_values["A$(i - 1)"]

        grads["W$i"] = 1 / nsamples * dZ * transpose(last_output)
        grads["B$i"] = 1 / nsamples * sum(dZ, dims = 2)
    end

    return grads
end

get_nlayers(params::Dict)::Int = (keys(params) |> length) / 2

function _update_weights!(
    params::Dict{String,T},
    grads::Dict{String,T},
    learning_rate,
)::Nothing where {T}
    nlayers = get_nlayers(params)

    for i = 1:nlayers
        params["W$i"] .-= learning_rate * grads["W$i"]
        params["B$i"] .-= learning_rate * grads["B$i"]
    end
end

end

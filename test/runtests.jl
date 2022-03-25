using MiraculousNN
using Test

using MLDatasets

@testset "Basic neural net workflow" begin
    ninstances = 1000
    nclasses = 10

    x, y_orig = MNIST.traintensor(Float64, 1:ninstances), MNIST.trainlabels(1:ninstances)

    x = reshape(x, (:, ninstances))
    y = MiraculousNN.utils.to_categorical(y_orig, nclasses)

    input_size = size(x, 1)
    output_size = size(y, 1)

    params = MiraculousNN._init_layers([input_size, 20, 10, output_size])

    @testset "init_layers" begin
        @test keys(params) |> length == 6

        @test size(params["W1"]) == (20, 784)
        @test size(params["B1"]) == (20, 1)

        @test size(params["W2"]) == (10, 20)
        @test size(params["B2"]) == (10, 1)

        @test size(params["W3"]) == (output_size, 10)
        @test size(params["B3"]) == (output_size, 1)

        @test params["W1"] != params["W2"]
        @test params["W1"] != params["W3"]

        @test params["B1"] != params["B2"]
        @test params["B1"] != params["B3"]
    end

    nn_values = MiraculousNN._forward(x, params)

    @testset "forward" begin
        @test keys(nn_values) |> length == 6

        @test size(nn_values["Z1"]) == (20, ninstances)
        @test size(nn_values["A1"]) == (20, ninstances)

        @test size(nn_values["Z2"]) == (10, ninstances)
        @test size(nn_values["A2"]) == (10, ninstances)

        @test size(nn_values["Z3"]) == (output_size, ninstances)
        @test size(nn_values["A3"]) == (output_size, ninstances)
    end

    grads = MiraculousNN._backward(params, nn_values, x, y)

    @testset "backward" begin
        @test keys(grads) |> length == 6

        @test size(grads["W1"]) == (20, 784)
        @test size(grads["B1"]) == (20, 1)

        @test size(grads["W2"]) == (10, 20)
        @test size(grads["B2"]) == (10, 1)

        @test size(grads["W3"]) == (output_size, 10)
        @test size(grads["B3"]) == (output_size, 1)
    end

    MiraculousNN._update_weights!(params, grads, 0.1)

    @testset "update_weights" begin
        @test keys(params) |> length == 6

        @test size(params["W1"]) == (20, 784)
        @test size(params["B1"]) == (20, 1)

        @test size(params["W2"]) == (10, 20)
        @test size(params["B2"]) == (10, 1)

        @test size(params["W3"]) == (output_size, 10)
        @test size(params["B3"]) == (output_size, 1)

        @test params["W1"] != params["W2"]
        @test params["W1"] != params["W3"]

        @test params["B1"] != params["B2"]
        @test params["B1"] != params["B3"]
    end
end


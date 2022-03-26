module forprop

relu(Z) = max.(0, Z)

softmax(Z) = exp.(Z) ./ sum(exp.(Z), dims = 1)

sigmoid(Z) = 1 / (1 + exp.(-Z))

end

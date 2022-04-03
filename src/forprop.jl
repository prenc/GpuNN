module forprop

relu(Z) = max(0, Z)

function softmax(Z) 
    exped = exp.(Z)
    return exped ./ sum(exped, dims = 1)
end

sigmoid(Z) = 1 / (1 + exp(-Z))

end

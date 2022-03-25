module backprop

function softmax(dA::Matrix, A::Matrix)::Matrix
    return dA - A
end

function relu(dA::Matrix, A::Matrix)::Matrix
    return dA .* (A .>= 0)
end

end

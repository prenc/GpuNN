module forprop

relu(mat::Matrix) = max.(0, mat)

softmax(mat::Matrix) = exp.(mat) ./ sum(exp.(mat), dims = 1)

end

module utils

function to_categorical(a::Vector{T}, num_classes)::Matrix{T} where {T<:Integer}
    num_instances = length(a)
    cat_a = zeros(T, num_classes, num_instances)

    for (class, col) in zip(arr, eachcol(cat_a))
        col[class + 1] = 1  # stupid as hell, needs to be reworked
    end

    return cat_a
end

end

module QMax

using LinearAlgebra
using Statistics
using Debugger

export oracle

"""
Quadratic maximum primal oracle. Describes the function:
f(x) =  max{ x_i^2 }
Arguments:
`x`: input value (the weights)
`svr`: SVR configuration struct instance
Outputs: the function value and the subgradient in `x`
"""
function oracle(x)
  n = length(x)
  loss, idx = findmax(x.^2)
  kronecker_delta = zeros(n)
  kronecker_delta[idx] = 1
  grad = 2 * (x .* kronecker_delta)
  return loss, grad
end

end

module JbSVR

using SVR
using LinearAlgebra
using Statistics
using JuMP
using Ipopt
using Debugger

# using ..Utils
import ..Utils
using ..Callbacks

include("BundleMethod.jl")
using .BundleMethod

export KnownSVR, predict, RBF_kernel, 
       Laplacian_kernel, Gaussian_kernel, kernel, SVR_conf


"""
SVR main struct
Arguments: 
`X`: train instances
`Y`: single output ground truth
`kernel_fun`: one among RBF_kernel/Laplacian_kernel/Gaussian_kernel
`ε_tube`: epsilon tube thikness
`C`: quadratic penality term
`kernel_args`: kernel keyword arguments. See the documentation of the kernel in use
"""
mutable struct SVR_conf
  m::Int
  n::Int
  X::Array{Float64}
  Y::Array{Float64}
  K::Array{Float64, 2}
  k::Function
  gramian::Function
  ε_tube::Float64
  C::Float64
  β::Union{Array{Float64}, Nothing}
  b::Union{Float64, Nothing}

  function SVR_conf(X, Y, kernel_fun; ε_tube=1e-2, C=2., kernel_args...)
    m, n = size(X)
    K = kernel(kernel_fun, X; kernel_args...)

    k = (x1, x2) -> kernel_fun(x1, x2; kernel_args...)
    gramian = (x) -> kernel(kernel_fun, X; X2=x, kernel_args...)
    new(m, n, X, Y, K, k, gramian, ε_tube, C, nothing, nothing)
  end
end


"""
Train an SVR struct
Arguments:
`svr`: svr structure configuration
`bundle_method_args`: bundle method keyword arguments. See the documentation of `bundle_method_cycle`
Outputs: a tuple containing the trained svr configuration, the optimum of f(x), 
         the termination status of the optimization procedure, the time elapsed
"""
function train(svr; bundle_method_args...)
  # Training ..... #
  x_zero = rand(svr.m + 1)
  bundle_oracle = (xi)->JbSVR.SVR_primal(xi, svr)
  (svr_params, f_star, infolog), time_elapsed = @timed BundleMethod.bundle_method_cycle(x_zero, bundle_oracle; bundle_method_args...)

  svr.β = svr_params[1:svr.m]
  svr.b = svr_params[svr.m+1]
  return svr, (infolog, time_elapsed)
end 


# ----- ----- SVR ORACLES ------ ---- #
# ----------- ----------- ----------- #
"""
SVR primal oracle
Arguments:
`x`: input value (the weights)
`svr`: SVR configuration struct instance
Outputs: the function value and the subgradient in `x`
"""
function SVR_primal(x, svr)
  β = x[1:svr.m]
  b = x[svr.m+1]

  # function ..... #
  residual = (svr.K' * β) .+ b .- svr.Y
  overflow = (residual.^2) .- svr.ε_tube
  δ = map(x -> if (x <= 0) 0 else 1 end, overflow)
  ε_loss = dot(overflow .* δ, ones(svr.m))
  tikhonov =  β' * β
  loss = (tikhonov + (svr.C * ε_loss)) / 2

  # gradient ..... #
  dloss = svr.K * (δ .* residual)
  dtikh = β
  sub_grad = dtikh .+ (svr.C .* dloss)
  b_grad = svr.C * dot(δ, residual)

  return loss, vcat(sub_grad, b_grad)
end


# ----- ----- SVR MISCELLANEOUS ------ ---- #
# ----------- ----------------- ----------- #
"""
Predicts the value of a set of samples. Caching is possible by passing 
an already computed gramian matrix
Arguments:
`svr`: svr configuration struct instance
`X`: a set of samples with shape: m rows, n columns (samples/features)
`gramian`: it is possible to take advantage of caching to speed up a prediction which 
           has to be made multiple times on the same data. Just compute the gramian 
           matrix using `svr.gramian` and invoke this function with that matrix
``
"""
function predict(svr; X=nothing, gramian=nothing)
  if isnothing(gramian)
    gramian = svr.gramian(X)
  end

  return (gramian' * svr.β) .+ svr.b
end


# ----- ----- SVR KERNELS ------ ---- #
# ----------- ----------- ----------- #
function RBF_kernel(x1, x2; σ::Real)
  σs = 2 * (σ^2)
  return exp(-norm(x1 - x2)^2 / σs)
end

function Laplacian_kernel(x1, x2; σ::Real)
    return exp(-sum((x1 - x2).^2 )^.5 / σ)
end

function Gaussian_kernel(x1, x2; kwargs...)
  return exp(-norm(x1 .- x2)^2)
end

"""
Computes the kernel matrix between two set of samples
Arguments: 
`kernel_fun`: one among RBF_kernel/Laplacian_kernel/Gaussian_kernel
`X{1/2}`: matrix of samples with shape: m_{1/2} rows, n_{1/2} columns (samples/features). 
          if X2 is omitted, X2 is assumed to be equal to X1
`kernel_args`: kernel function arguments
Outputs: a matrix with shape (m_1, m_2) with: `K_ij = kernel_fun(X1_i, X2_j)`
"""
function kernel(kernel_fun, X1; X2=nothing, kernel_args...)
  if isnothing(X2)
    X2 = X1
  end
  
  m = size(X1)[1]
  n = size(X2)[1]
  K = fill(0.0, m, n)

  for i = 1:m
      for j = 1:n
        K[i, j] = kernel_fun(X1[i, :], X2[j, :]; kernel_args...)
      end
  end

  return K
end


# ----- ----- OFF-THE-SHELF SVR ------ ---- #
# ----------- ----------------- ----------- #
function KnownSVR(X, Y, kernel)
  if kernel == Utils.KERNEL_RBF
    k_func = SVR.RBF
  elseif kernel == Utils.KERNEL_LINEAR
    k_func = SVR.LINEAR
  else
    k_func = SVR.POLY
  end
  pmodel = SVR.train(Y, permutedims(X); kernel_type=k_func)
  y_pr = SVR.predict(pmodel, permutedims(X));
  # SVR.savemodel(pmodel, "mg.model")
  # SVR.freemodel(pmodel)

  l2_y = (Y .- y_pr).^2
  l2_y = mean(l2_y)
  return l2_y, pmodel
end

end


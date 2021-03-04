using DynamicalSystems

# rewrite the problems for DynamicalSystems as 1d problems


#Δ2 = Δ_PBC(n, dx)

function conv_2d_to_1d_array(array_in::AbstractArray{T,2}) where T<:Number
    N = size(array_in,1)
    array_out = zeros(eltype(array_in), 2N)
    array_out[1:N] = array_in[:,1]
    array_out[N+1:end] = array_in[:,2]
    array_out
end

function conv_2d_to_1d_array(array_in::AbstractArray{T,3}) where T<:Number
    N = size(array_in,1)
    N_t = size(array_in,3)
    array_out = zeros(eltype(array_in), 2N, N_t)
    array_out[1:N,:] = array_in[:,1,:]
    array_out[N+1:end,:] = array_in[:,2,:]
    array_out
end

function conv_1d_to_2d_array(array_in::AbstractArray{T,1}) where T<:Number
    reshape(array_in, (:,2))
end

function conv_1d_to_2d_array(array_in::AbstractArray{T,2}) where T<:Number
    N, N_t = size(array_in)
    n = Int(N/2)
    array_out = zeros(eltype(array_in), n, 2, N_t)
    array_out[:,1,:] = array_in[1:n,:]
    array_out[:,2,:] = array_in[n+1:end,:]
    array_out
end

"""
function f_analytic!(du, u, p, t)
    ReU = @view u[1:Ndim]
    ImU = @view u[Ndim+1:end]
    α, β = p[1], complex(0,p[2])

    du[1:Ndim] .= -Δ2*(ReU .- α.*ImU)
    du[Ndim+1:end] .= -Δ2*(ImU .+ α.*ReU)

    for i=1:Ndim
        resnonlin = non_lin(complex(ReU[i], ImU[i]), β)

        du[i] += real(resnonlin)
        du[i+Ndim] += imag(resnonlin)
    end
end
"""

"""
function f_npde(u, p, t)
    ReU = @view u[1:Ndim]
    ImU = @view u[Ndim+1:end]
    nn_res = re_nn(p)(transpose(reshape(u,Ndim,2)))
    return (Δ2*(ReU - α.*ImU))*matRe + (Δ2*(ImU + α.*ReU))*matIm + transpose(nn_res)
end

# test if the equations are really the same
function compare_probs(prob, f2)
    sol1 = Array(solve(prob, Tsit5(), saveat=dt))
    sol2 = Array(solve(ODEProblem(f2, prob.u0, prob.tspan, prob.p), Tsit5(), saveat=dt))

    if (sol1 == sol2)
        return true
    else
        return sol1, sol2
    end
end

if !(compare_probs(prob_reim, f_analytic!))
    error("Problems for lyapunov computation not equal")
end

if !(compare_probs(prob, f_npde))
    error("Problems for lyapunov computation not equal")
end


function lyapunov_analytic(prob; N=5, k=1)
    if !(compare_probs(prob, f_analytic!))
        error("Theres something wrong with the convert to DynamicalSystems.jl")
    end

    prob_new = ODEProblem(f_analytic!, conv_2d_to_1d_array(prob.u0), prob.tspan, prob.p)
    lyapunov_exps(prob_new, N=N, k=k)
end

function lyapunov_npde(prob; N=5, k=1)
    if !(compare_probs(prob, f_npde))
        error("Theres something wrong with the convert to DynamicalSystems.jl")
    end

    prob_new = ODEProblem(f_npde, conv_2d_to_1d_array(prob.u0), prob.tspan, prob.p)
    lyapunov_exps(prob_new, N=N, k=k)
end

"""

function lyapunov_exps(prob; N=5, k=1)
    ds = ContinuousDynamicalSystem(prob.f, prob.u0, prob.p)
    if k==1
        lyapunov(ds, 200.,Ttr=20.)
    else
        lyapunovs(ds,N,k)
    end
end

function forecast_horizon(prediction::AbstractArray{T,2}, truth::AbstractArray{T,2}, abstol=1e-1) where T<:Number


    δ = abs.(prediction .- truth)

    N_δ = (sum(δ .> abstol, dims=1)/size(truth,1))[:]

    (findfirst(N_δ .> 0), N_δ)
end

forecast_horizon(prediction::AbstractArray{T,3}, truth::AbstractArray{T,3}, abstol=1e-1) where T<:Number = forecast_horizon(reshape(prediction,(:,size(prediction,3))), reshape(truth,(:,size(prediction,3))), abstol)


function forecast_δ(prediction::AbstractArray{T,2}, truth::AbstractArray{T,2}, mode::String="both") where T<:Number

    if !(mode in ["mean","largest","both","norm"])
        error("mode has to be either 'mean', 'largest' or 'both', 'norm'.")
    end

    δ = abs.(prediction .- truth)

    if mode == "mean"
        return mean(δ, dims=1)
    elseif mode == "largest"
        return maximum(δ, dims=1)
    elseif mode == "norm"
        return sqrt.(sum((prediction .- truth).^2, dims=1))./sqrt.(sum(truth.^2, dims=1))
    else
        return (mean(δ, dims=1), maximum(δ, dims=1))
    end
end
forecast_δ(prediction::AbstractArray{T,3}, truth::AbstractArray{T,3}, mode="both") where T<:Number = forecast_δ(reshape(prediction,(:,size(prediction,3))), reshape(truth,(:,size(prediction,3))), mode)

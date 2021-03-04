# finite difference version

begin
    using Plots
    using OrdinaryDiffEq
    using JLD2
    using BlockArrays
    using LinearAlgebra
    using SparseArrays
end

begin
    using Random
    Random.seed!(1234)

    LOAD_DATA = false
    SAVE_DATA = false
    PLOT_CG = false
end

if !(@isdefined n)
    n = 50
end

if !(@isdefined L)
    L = 75
end

if !(@isdefined dt)
    dt = 0.1
end

if !(@isdefined N_t)
    N_t = 600
end

begin
    #n = 32 # n = 50
    #L = 50 # L = 75
    α = 2.0im
    β = -1.0im
    dx = L/n
    u0 = 0.01*(rand(ComplexF64, (n,n)) .- (0.5 + 0.5im))
    non_lin(x, β) = x - (1. + β)*abs(x)^2*x
end


if !(@isdefined COMPUTE_DATA)
    COMPUTE_DATA = !(LOAD_DATA)
end

array_to_1d(A) = reshape(A, (:,N_t))
function Δ_PBC(n::Int, dx::Real=1.) # this is for periodic BC, this means that there are some extra -1 in there (in comparision to Diriclet)
    Δ = BlockArray(zeros(n^2,n^2), n*ones(Int,n), n*ones(Int,n))
    inner_block = diagm(0=>4*ones(n), 1=>(-1*ones(n-1)), -1=>(-1*ones(n-1)), (n-1)=>(-1*ones(1)), (-n+1)=>(-1*ones(1))) ./ (dx^2)
    minus_block = diagm(0=>(-1*ones(n))) ./ (dx^2)
    for i=1:n
        Δ[Block(i,i)] .= inner_block
    end
    Δ[Block(1,2)] .= minus_block
    Δ[Block(1,n)] .= minus_block
    for i=2:n-1
        Δ[Block(i,i-1)] .= minus_block
        Δ[Block(i,i+1)] .= minus_block
    end
    Δ[Block(n,1)] .= minus_block
    Δ[Block(n,n-1)] .= minus_block
    sparse(Δ) # this will make the multiplication much much faster, blockbandedmatrix could probably even improve this
end

array_to_complex(A::AbstractArray{T,2}) where T<:Real = complex.(A[:,1], A[:,2])

array_to_complex(A::AbstractArray{T,3}) where T<:Real = complex.(A[:,1,:], A[:,2,:])

function split_reim(A::AbstractArray{T,2}) where T<:Complex
    B = zeros(eltype(real.(A)),size(A)[1], 2, size(A)[2])
    B[:,1,:] .= real.(A)
    B[:,2,:] .= imag.(A)
    B
end

function split_reim(A::AbstractArray{T,1}) where T<:Complex
    B = zeros(eltype(real.(A)),length(A), 2)
    B[:,1] .= real.(A)
    B[:,2] .= imag.(A)
    B
end

u0 = reshape(u0, (:,))
N = size(u0, 1)
Δ = Δ_PBC(n, dx)
u0_reim = split_reim(u0)

const Ndim = N

function cgle_fd!(du, u, p, t)
    α, β = p
    du .= -(1 .+ α).*(Δ*u) .+ non_lin.(u, β)
end

function cgle_fd_reim!(du, u, p, t)
    ReU = @view u[:,1]
    ImU = @view u[:,2]
    α, β = p[1], complex(0,p[2])

    du[:,1] .= -Δ*(ReU .- α.*ImU)
    du[:,2] .= -Δ*(ImU .+ α.*ReU)

    for i=1:Ndim
        resnonlin = non_lin(complex(ReU[i], ImU[i]), β)

        du[i,1] += real(resnonlin)
        du[i,2] += imag(resnonlin)
    end
end

t_start = 200.0
t_end = t_start + N_t*dt

pars = [α; β]

prob_reim = ODEProblem(cgle_fd_reim!, Float32.(u0_reim), (0.,t_end), [imag(α); imag(β)])

#throw away transient
println("solving....")
if LOAD_DATA
    @load "ginsburg-data.jld2" dat
else
    if COMPUTE_DATA


        dat_reim = Array(solve(prob_reim, Tsit5(), saveat=t_start:dt:t_end))

        if SAVE_DATA
            @save "ginsburg-data.jld2" dat
        end
    end
end

if PLOT_CG
    Plots.pyplot()
    anim = Plots.@animate for i in eachindex(t)
        Plots.heatmap(abs.(reshape(array_to_complex(dat_reim[:,:,i]),(32,32,:)))[:,:,1], clim=(-1.5,1.5))
    end

    gif(anim, "clge2d-3232-short.gif")
end

if LYAPUNOV

    function cgle_fd_reim2!(du, u, p, t)
        ReU = @view u[1:Ndim]
        ImU = @view u[Ndim:end]
        α, β = p[1], complex(0,p[2])

        du[1:Ndim] .= -Δ*(ReU .- α.*ImU)
        du[Ndim:end] .= -Δ*(ImU .+ α.*ReU)

        for i=1:Ndim
            resnonlin = non_lin(complex(ReU[i], ImU[i]), β)

            du[i] += real(resnonlin)
            du[i+Ndim] += imag(resnonlin)
        end
    end
    u0_reim = [real.(u0); imag.(u0)]

    function jac(J,u,p,t)
    (-(1 .+ α).*Δ) + Diagonal( 1. .- (1 - β)*(2.0*abs.(u).*u + abs.(u).^2))
    end

    using DynamicalSystems

    ds = ContinuousDynamicalSystem(cgle_fd_reim2!, u0_reim, [imag(α), imag(β)])

    println(lyapunov(ds,500.,Ttr=100.))

end

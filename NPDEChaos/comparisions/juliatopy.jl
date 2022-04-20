using BlockArrays
using LinearAlgebra
using SparseArrays
using OrdinaryDiffEq
using Random
Random.seed!(1234)

function LaplacianCGLE(n::Int, dx::Real=1.) # this is for periodic BC, this means that there are some extra -1 in there (in comparision to Diriclet)
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


if !(@isdefined n)
    const n = 50
end

if !(@isdefined L)
    const L = 75
end

if !(@isdefined dt)
    const dt = 0.1
end

if !(@isdefined N_t)
    const N_t = 3000
end
const Ndim = n*n


lap = LaplacianCGLE(n, L/n)


array_to_1d(A) = reshape(A, (:,N_t))

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





function cgle_fd_reim!(du, u, p, t)
    ReU = @view u[:,1]
    ImU = @view u[:,2]
    α, β = p[1], complex(0,p[2])

    du[:,1] .= -lap*(ReU .- α.*ImU)
    du[:,2] .= -lap*(ImU .+ α.*ReU)

    for i=1:Ndim
        resnonlin = non_lin(complex(ReU[i], ImU[i]), β)

        du[i,1] += real(resnonlin)
        du[i,2] += imag(resnonlin)
    end
end


non_lin(x, β) = x - (1. + β)*abs(x)^2*x

function compute_data()

    begin
        #n = 32 # n = 50
        #L = 50 # L = 75
        α = 2.0im
        β = -1.0im
        #u0 = 0.1*(rand(ComplexF64, (n,n)) .- 0.5)
        dx = L/n
        u0 = 0.01*(rand(ComplexF64, (n,n)) .- (0.5 + 0.5im))
        #u0 = (im.*x .+ y) .* exp.(-0.03*(x.^2 .+ y.^2)) .+
    end


    u0 = reshape(u0, (:,))
    N = size(u0, 1)

    u0_reim = split_reim(u0)



    t_start = 200.0
    t_end = t_start + N_t*dt

    pars = [α; β]

    #prob = ODEProblem(cgle_fd!, u0, (0.,t_end), pars)
    prob_reim = ODEProblem(cgle_fd_reim!, Float32.(u0_reim), (0.,t_end), [imag(α); imag(β)])

    #throw away transient
    println("solving....")

    dat_reim = Array(solve(prob_reim, Tsit5(), saveat=t_start:dt:t_end))

    dat = zeros(eltype(dat_reim),2N,size(dat_reim,3))
    dat[1:N,:] = dat_reim[:,1,:]
    dat[N+1:end,:] = dat_reim[:,2,:]

    return dat
end

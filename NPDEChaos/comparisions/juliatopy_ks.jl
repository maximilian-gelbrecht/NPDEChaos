# finite difference version

begin
    using OrdinaryDiffEq
    using LinearAlgebra
    using SparseArrays
end

begin
    using Random
    Random.seed!(1234)
end

if !(@isdefined n)
    n = 128
    #n = 2048
end

if !(@isdefined L)
    L = 35
    #L = 560
end

if !(@isdefined dt)
    dt = 0.1
end

if !(@isdefined N_t)
    N_t = 5000
end


function ks_fd_dx1(n, dx)
    ∂x = (diagm(1=>ones(n-1)) + diagm(-1=>-1*ones(n-1)))
    ∂x[1,end] = -1
    ∂x[end,1] = 1
    ∂x ./= (2*dx)
    ∂x = sparse(∂x)
    return ∂x
end

function ks_fd_dx2(n, dx)
    ∂x2 = diagm(0=>-2*ones(n)) + diagm(-1=>ones(n-1)) + diagm(1=>ones(n-1))
    ∂x2[1,end] = 1
    ∂x2[end,1] = 1
    ∂x2 ./= (dx)^2
    ∂x2 = sparse(∂x2)
    return ∂x2
end

function ks_fd_dx4(n, dx)
    ∂x2 = ks_fd_dx2(n,dx)
    return sparse(∂x2*∂x2)
end


function ks_fd_operators(n, dx)
    ∂x = (diagm(1=>ones(n-1)) + diagm(-1=>-1*ones(n-1)))
    ∂x[1,end] = -1
    ∂x[end,1] = 1
    ∂x ./= (2*dx)
    ∂x = sparse(∂x)
    ∂x2 = diagm(0=>-2*ones(n)) + diagm(-1=>ones(n-1)) + diagm(1=>ones(n-1))
    ∂x2[1,end] = 1
    ∂x2[end,1] = 1
    ∂x2 ./= (dx)^2
    ∂x2 = sparse(∂x2)
    ∂x4 = sparse(∂x2*∂x2)
    return ∂x, ∂x2, ∂x4
end

dx = L/n

∂x, ∂x2, ∂x4 = ks_fd_operators(n, dx)


function ks_oop(u,p,t)
        -∂x4*u - ∂x2*u - u.*(∂x*u)
end

function compute_ks_data()

    begin
        u0 = 0.01*(rand(Float32, n) .- 0.5)
        #u0 = (im.*x .+ y) .* exp.(-0.03*(x.^2 .+ y.^2)) .+
    end



    t_start = 200.0
    #t_start = 0.0
    t_end = t_start + N_t*dt





    #prob = ODEProblem(cgle_fd!, u0, (0.,t_end), pars)
    prob = ODEProblem(ks_oop, Float32.(u0), (0.,t_end), [])


    dat = Array(solve(prob, Tsit5(), saveat=t_start:dt:t_end))

    return dat
end

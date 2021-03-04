# finite difference version
"""
This script is called by the NPDE script to compute the training/validation data of the KS.

It can also be run alone to compute the maximum Lyapunov exponent (and the data) of the system.
"""

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
    PLOT = false
    LYAPUNOV = false
end

if !(@isdefined n)
    n = 64
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
    N_t = 2000
end

begin
    dx = L/n
    u0 = 0.01*(rand(Float32, n) .- 0.5)
end

if !(@isdefined COMPUTE_DATA)
    COMPUTE_DATA = !(LOAD_DATA)
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

∂x, ∂x2, ∂x4 = ks_fd_operators(n, dx)

t_start = 200.0
t_end = t_start + N_t*dt

function ks_oop(u,p,t)
        -∂x4*u - ∂x2*u - u.*(∂x*u)
end

prob = ODEProblem(ks_oop, Float32.(u0), (0.,t_end))

#throw away transient
println("solving....")
if LOAD_DATA
    @load "ks-data.jld2" dat
else
    if COMPUTE_DATA

        dat = Array(solve(prob, Tsit5(), saveat=t_start:dt:t_end))

        if SAVE_DATA
            @save "ks-data.jld2" dat
        end
    end
end


if PLOT
    Plots.gr()
    heatmap(dat)
end

if LYAPUNOV
    using StaticArrays

    function ks_oop_vec(u,p,t)
            SVector{n}(-∂x4*u - ∂x2*u - u.*(∂x*u))
    end

    using DynamicalSystems
    ds = ContinuousDynamicalSystem(ks_oop_vec, prob.u0, prob.p)

    println(lyapunov(ds, 2000.,Tr=20.))
end



if LYAPUNOV
    using StaticArrays

    function ks_oop_vec(u,p,t)
           SVector{n}(-∂x4*u - ∂x2*u - u.*(∂x*u))
   end



   jac_∂ = -∂x4 - ∂x2
   function ks_jac(u, p, t)
      jac_∂ - ((u.*∂x) + Diagonal((∂x*u)))
end



    using DynamicalSystems
    ds = ContinuousDynamicalSystem(ks_oop_vec, prob.u0, prob.p, ks_jac)
    #println(lyapunovs(ds, N=1, k=1, Ttr=100.))

    println(lyapunov(ds, 500.,Tr=100.))
end

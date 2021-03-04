import Flux.trainable
using Flux


function indexing_3d_to_2d(Nx,Ny,Nz,ix,iy,iz)
  Nxy = Nx*Ny
  ind_mat = reshape(1:Nxy,Nx,Ny)
  [reshape(ind_mat[ix,iy],:),iz]
end


"""
A version of Dense that's fully GPU compatible for batched data
"""
struct DenseGPU{F,S,T}
  W::S
  b::T
  σ::F
end

DenseGPU(W, b) = DenseGPU(W, b, identity)

function DenseGPU(in::Integer, out::Integer, σ = identity;
               initW = Flux.glorot_uniform, initb = zeros)
  return DenseGPU(initW(out, in), initb(out), σ)
end

Flux.@functor DenseGPU

function (a::DenseGPU)(x::AbstractArray)
  W, b, σ = a.W, a.b, a.σ
  σ.(W*x + b*P)
end


"""
    ParSkipConnection(layer, connection)

A version of SkipConnection that is paramaterized with one single parameter, thus the output is `connection(w .* layer(input), input)`
"""
struct ParSkipConnection{F}
  layers
  connection  #user can pass arbitrary connections here, such as (a,b) -> a + b
  w::F
end

ParSkipConnection(layer, connection; initw=Flux.glorot_uniform) = ParSkipConnection(layer, connection, initw())

Flux.@functor ParSkipConnection

function (skip::ParSkipConnection)(input)
  skip.connection(skip.w .* skip.layers(input), input)
end

function Base.show(io::IO, b::ParSkipConnection)
  print(io, "ParSkipConnection(", b.layers, ", ", b.connection, ", ,w=",b.w,")")
end

"""

  NablaSkipConnection1

A version of the `NablaSkipConnection` that hard bounds the parameter to the interval `[0,1]`.

"""
struct NablaSkipConnection1{F}
  w::F
end
NablaSkipConnection1() = NablaSkipConnection1(rand(Float32))

function (skip::NablaSkipConnection1{F})(input) where F <: Number
  if skip.w < 0f0
    w = F(0)
  elseif skip.w > 1f0
    w = F(1)
  end

  skip.w .* (∂x * input) + (F(1) - skip.w) * input
end

Flux.@functor NablaSkipConnection1

"""

  NablaSkipConnection2

A version of the `NablaSkipConnection` where the parameter is not hard bound but is penalized later in the loss function to be within `[0,1]`

"""
struct NablaSkipConnection2{F}
  w::F
  one::F
end

function NablaSkipConnection2(w::F, gpu::Bool=false) where F<:Number
  if gpu
    return NablaSkipConnection2(CuArray([w]), CuArray([F(1)]))
  else
    return NablaSkipConnection2([w],[F(1)])
  end
end

NablaSkipConnection2(w::AbstractArray{T,1}) where T<:Number = NablaSkipConnection2(w, [T(1)])

NablaSkipConnection2(w::CuArray{T,1}) where T<:Number= NablaSkipConnection2(w, CuArray([T(1)]))


function (skip::NablaSkipConnection2)(input)
  skip.w .* (∂x * input) + (skip.one .- skip.w) .* input
end

Flux.@functor NablaSkipConnection2 (w,)

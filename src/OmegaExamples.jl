module OmegaExamples

using Reexport 

include("CarCrash.jl")
@reexport using .CarCrash

include("CarCrash3D.jl")
@reexport using .CarCrash3D

end
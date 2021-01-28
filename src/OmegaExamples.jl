module OmegaExamples

using Reexport 

include("CarCrash/CarCrash.jl")
@reexport using .CarCrash

end
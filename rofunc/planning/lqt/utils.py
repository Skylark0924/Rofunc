# param = {
#     "nbData": 500,  # Number of data points
#     "nbPoints": 10,  # Number of viapoints
#     "nbVarPos": 7,  # Dimension of position data
#     "nbDeriv": 2,  # Number of static and dynamic features (2 -> [x,dx])
#     "dt": 1e-2,  # Time step duration
#     "rfactor": 1e-8  # Control cost
# }
# param["nb_var"] = param["nbVarPos"] * param["nbDeriv"]  # Dimension of state vector
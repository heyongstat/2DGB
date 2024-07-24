# 2DGB
Two-way dimension reduction for well-structured matrix-valued data is growing popular in the past few years. To
achieve robustness against individual matrix outliers with large spikes, arising either from heavy-tailed noise or large
individual low-rank signals deviating from the population subspace, we frst calculate the leading singular subspaces
of each individual matrix, and then fnd the barycenter of the locally estimated subspaces across all observations, in
contrast to the existing methods which frst integrate data across observations and then do eigenvalue decomposition.
In addition, a robust cut-oﬀ dimension determination criteria is suggested based on comparing the eigenvalue ratios
of the corresponding Euclidean means of the projection matrices. Theoretical properties of the resulting estimators
are investigated under mild conditions. Thorough numerical simulation studies justify the advantages and robustness
of the proposed methods over the existing tools. Two real examples associated with medical imaging and fnancial
portfolios are given to provide empirical evidences on the arguments in this work and also the usefulness of our
algorithms.

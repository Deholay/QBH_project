import numpy as np

def calculate_edit_distance(d, query_diff, target_diff):
    """
    Calculate the edit distance between query_diff and target_diff using dynamic programming.
    """
    M = len(query_diff)
    N = len(target_diff)

    # Initialize (M+1) x (N+1) matrix D
    D = np.zeros((M+1, N+1))

    # Set base cases
    for m in range(M+1):
        D[m][0] = m * d  # D(n,0) = d * n
    for n in range(N+1):
        D[0][n] = 0      # D(0,m) = 0

    # Fill the matrix
    for m in range(1, M+1):
        for n in range(1, N+1):
            dist = abs(query_diff[m-1] - target_diff[n-1])  # dist(X(m), Y(n))
            D[m][n] = min(
                D[m-1][n] + d,       # Insertion
                D[m][n-1] + d,       # Deletion
                D[m-1][n-1] + dist      # Substitution
            )

    # Calculate the edit distance as the minimum of the last row
    edit_distance = np.min(D[M, :])
    return edit_distance, D
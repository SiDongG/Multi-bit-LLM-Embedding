"""
ldpc_codec.py
A lightweight LDPC encoder/decoder module using Sum-Product BP.

USAGE:
    from ldpc_codec import LDPCCode
    code = LDPCCode(H)
    cw = code.encode(message_bits)
    decoded = code.decode(received_cw, max_iters=50)
"""

import numpy as np


# ==========================================================
#   Utility: GF(2) Gaussian elimination to build generator G
# ==========================================================
def gf2_rref(A):
    """Return RREF of A over GF(2), and pivot columns."""
    A = A.copy() % 2
    m, n = A.shape
    pivots = []
    r = 0
    for c in range(n):
        # find pivot row
        pivot = np.where(A[r:, c] == 1)[0]
        if len(pivot) == 0:
            continue
        i = pivot[0] + r
        # swap
        A[[r, i]] = A[[i, r]]
        pivots.append(c)
        # eliminate below
        for i in range(r+1, m):
            if A[i, c] == 1:
                A[i] ^= A[r]
        r += 1
        if r == m:
            break

    # eliminate above pivots
    for ri, c in enumerate(pivots):
        for i in range(ri):
            if A[i, c] == 1:
                A[i] ^= A[ri]
    return A, pivots


def build_generator_from_H(H):
    """
    Given parity-check matrix H (m × n),
    build generator matrix G for a systematic code.

    H = [A | B]  → find G such that H G^T = 0.
    We perform RREF and extract pivot structure.
    """
    m, n = H.shape
    H_rref, pivots = gf2_rref(H)

    pivots = list(pivots)
    free = [i for i in range(n) if i not in pivots]

    k = len(free)  # number of information bits
    G = np.zeros((k, n), dtype=np.uint8)

    for row_idx, j in enumerate(free):
        G[row_idx, j] = 1
        for (piv_row, piv_col) in enumerate(pivots):
            if H_rref[piv_row, j] == 1:
                G[row_idx, piv_col] = 1
    return G


# ==========================================================
#   LDPC Code Class
# ==========================================================
class LDPCCode:
    def __init__(self, H):
        """
        H: parity-check matrix (numpy array 0/1)
        """
        self.H = H.astype(np.uint8)
        self.m, self.n = H.shape

        self.G = build_generator_from_H(self.H)
        self.k = self.G.shape[0]

        # Precompute adjacency lists
        self.var_nodes = [np.where(self.H[:, j] == 1)[0] for j in range(self.n)]
        self.check_nodes = [np.where(self.H[i, :] == 1)[0] for i in range(self.m)]

    # ------------------------------------------------------
    # Encoding: c = u G  (mod 2)
    # ------------------------------------------------------
    def encode(self, u):
        """
        Encode message u (length k) into a codeword (length n).
        u: numpy array of 0/1 bits
        """
        u = np.array(u, dtype=np.uint8)
        assert len(u) == self.k
        cw = (u @ self.G) % 2
        return cw.astype(np.uint8)

    # ------------------------------------------------------
    # Sum-Product Belief Propagation Decoder
    # channel_llr: log( P(y|x=0) / P(y|x=1) )
    # BP on Tanner graph
    # ------------------------------------------------------
    def decode(self, received_bits, max_iters=50, p_error=0.05):
        """
        received_bits: array of 0/1 (after channel)
        p_error: assumed independent bit-flip prob → used to compute LLR
        """
        y = np.array(received_bits, dtype=np.uint8)

        # channel LLR (binary symmetric channel)
        llr0 = np.log((1 - p_error) / p_error)
        channel_llr = np.where(y == 0, llr0, -llr0).astype(float)

        # messages: msg_check_to_var[i][j], msg_var_to_check[j][i]
        msg_v_to_c = {}  # (var j → check i)
        msg_c_to_v = {}  # (check i → var j)

        # init messages
        for j in range(self.n):
            for i in self.var_nodes[j]:
                msg_v_to_c[(j, i)] = channel_llr[j]

        for it in range(max_iters):

            # ---- CHECK NODE UPDATE ----
            for i in range(self.m):
                connected = self.check_nodes[i]
                for j in connected:
                    # product of tanh of half incoming messages
                    prod = 1.0
                    for jp in connected:
                        if jp == j: continue
                        prod *= np.tanh( msg_v_to_c[(jp, i)] / 2 )
                    msg_c_to_v[(i, j)] = 2 * np.arctanh(prod)

            # ---- VARIABLE NODE UPDATE ----
            for j in range(self.n):
                connected = self.var_nodes[j]
                for i in connected:
                    total = channel_llr[j]
                    for ip in connected:
                        if ip == i: continue
                        total += msg_c_to_v[(ip, j)]
                    msg_v_to_c[(j, i)] = total

            # ---- Marginals ----
            marginals = np.zeros(self.n)
            for j in range(self.n):
                total = channel_llr[j]
                for i in self.var_nodes[j]:
                    total += msg_c_to_v[(i, j)]
                marginals[j] = total

            # Tentative hard decision
            hard = (marginals < 0).astype(np.uint8)

            # Check parity
            syndrome = (self.H @ hard) % 2
            if not syndrome.any():
                return hard  # success!

        return hard  # return last estimate


# ==========================================================
# Example small test (only runs if file is executed)
# ==========================================================
if __name__ == "__main__":
    # small parity-check example
    H = np.array([
        [1,1,0,1,0,0],
        [0,1,1,0,1,0],
        [1,0,1,0,0,1],
    ], dtype=np.uint8)

    code = LDPCCode(H)

    msg = np.array([1,0,1], dtype=np.uint8)
    cw = code.encode(msg)

    print("Message:", msg)
    print("Codeword:", cw)

    # introduce some noise
    rx = cw.copy()
    rx[1] ^= 1
    rx[4] ^= 1

    decoded = code.decode(rx, max_iters=30)
    print("Decoded:", decoded)

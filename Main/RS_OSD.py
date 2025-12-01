import numpy as np
import reedsolo
import itertools


# ================================================================
# RS CODEC INITIALIZATION (now dynamic)
# ================================================================
def init_rs_codec(n, k):
    """
    Build a Reed–Solomon codec with user-defined (n, k).
    Returns (rs_codec, parity_symbols, t_correctable).
    """
    parity = n - k
    rs = reedsolo.RSCodec(parity)
    t = parity // 2     # correctable symbol errors
    return rs, parity, t


# ================================================================
# RELIABILITY COMPUTATION (dynamic bit-lengths + symbol count)
# ================================================================
def compute_symbol_reliability(segment_bits, n, k):
    """
    Convert segment-based bit reliabilities into symbol-level reliabilities.
    
    - RS(n, k) has n symbols (each 1 byte = 8 bits)
    - total bits = n * 8
    """
    TOTAL_BITS = n * 8
    BITS_PER_SYMBOL = 8

    reliability_bits = np.zeros(TOTAL_BITS, dtype=float)

    # Fill bit-wise reliability sequentially using segment lengths
    bit_pos = 0
    for seg_id, bits in segment_bits.items():
        L = len(bits)
        rel = 1.0 / L
        for _ in range(L):
            if bit_pos < TOTAL_BITS:
                reliability_bits[bit_pos] = rel
            bit_pos += 1

    # If segments do not fill all bits, remaining bits get reliability = 1.0
    if bit_pos < TOTAL_BITS:
        reliability_bits[bit_pos:TOTAL_BITS] = 1.0

    # Aggregate 8 bits → 1 symbol reliability
    symbol_rel = np.zeros(n)
    for i in range(n):
        start = i * BITS_PER_SYMBOL
        end   = (i + 1) * BITS_PER_SYMBOL
        symbol_rel[i] = np.sum(reliability_bits[start:end])

    return symbol_rel



# ================================================================
# TOP-K CANDIDATES FOR SEGMENTS
# ================================================================
def get_symbol_candidates_from_stats(segment_stats, symbol_index, bits_per_symbol=8, K=3):
    """
    Generate K^S candidate RS symbols for symbol_index, using only segment_stats.
    Automatically supports variable bit lengths.
    """

    total_bits = sum(info["L"] for info in segment_stats.values())
    start = symbol_index * bits_per_symbol
    end   = min((symbol_index + 1) * bits_per_symbol, total_bits)
    sym_len = end - start

    if start >= total_bits:
        return []

    # Identify which segments overlap this symbol
    contributing = []   # (seg_id, local_start, take_len)
    current_bitpos = 0

    for seg_id, info in segment_stats.items():
        L = info["L"]
        seg_start = current_bitpos
        seg_end   = current_bitpos + L

        overlap_start = max(seg_start, start)
        overlap_end   = min(seg_end,   end)

        if overlap_end > overlap_start:
            local_start = overlap_start - seg_start
            take_len    = overlap_end   - overlap_start
            contributing.append((seg_id, local_start, take_len))

        current_bitpos += L
        if current_bitpos >= end:
            break

    if not contributing:
        return []

    # Pull top-K bitstrings for each contributing segment
    seg_bit_cands = []
    for seg_id, local_start, take_len in contributing:
        counts = segment_stats[seg_id]["counts"]
        topK = sorted(counts.items(), key=lambda x: -x[1])[:K]

        slices = []
        for bits, cnt in topK:
            if cnt > 0 and len(bits) >= local_start + take_len:
                slices.append(bits[local_start : local_start + take_len])

        if not slices:
            return []

        seg_bit_cands.append(slices)

    # Cartesian product → K^S full symbol candidates
    candidates = []
    for combo in itertools.product(*seg_bit_cands):
        full_bits = "".join(combo)
        if len(full_bits) < bits_per_symbol:
            full_bits = full_bits.ljust(bits_per_symbol, "0")

        candidates.append(int(full_bits, 2))

    return candidates



# ================================================================
# ENCODING FOR RS(n, k)
# ================================================================
def rs_encode_bits(bits, n, k, rs):
    """
    Input:  (k * 8) bits
    Output: (n * 8) bits RS codeword
    """
    msg_bytes = np.packbits(bits).astype(np.uint8)          # k bytes
    encoded = rs.encode(msg_bytes.tobytes())                # n bytes
    return np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))



# ================================================================
# OSD-1 DECODER FOR RS(n, k)
# ================================================================
def rs_osd1_decode(rs, received_symbols, reliability_symbols, segment_stats, K=3, L=2):
    """
    OSD-1 with segment-based symbol candidates:
      - expand at L least reliable symbol positions
      - try top-K candidates per contributing segment
    """

    n = len(received_symbols)
    least_rel = np.argsort(reliability_symbols)

    candidates = [received_symbols.copy()]

    # Expand L worst positions
    for pos in least_rel[:L]:
        sym_cands = get_symbol_candidates_from_stats(
            segment_stats=segment_stats,
            symbol_index=pos,
            bits_per_symbol=8,
            K=K
        )

        if not sym_cands:
            continue

        new_list = []
        for base in candidates:
            for sym in sym_cands:
                mod = base.copy()
                mod[pos] = sym
                new_list.append(mod)

        candidates = new_list

    # Evaluate all candidates
    best_msg = None
    best_metric = float("inf")

    for cand in candidates:
        try:
            decoded = rs.decode(bytes(cand))[0]  # returns k-byte message
        except reedsolo.ReedSolomonError:
            continue

        # Weighted Hamming metric
        dist = sum(
            1.0 / (reliability_symbols[i] + 1e-9)
            for i in range(n)
            if cand[i] != received_symbols[i]
        )

        if dist < best_metric:
            best_metric = dist
            best_msg = decoded

    return best_msg



# ================================================================
# FULL DECODER (bits → RS codeword → OSD → message bits)
# ================================================================
def rs_decode_bits(received_bits, segment_bits, segment_stats, n, k, rs, K=2, L=3):
    """
    Generic RS(n, k) decoder using improved OSD-1.
    """
    received_bits = np.array(received_bits, dtype=np.uint8)

    if received_bits.size != n * 8:
        raise ValueError(f"Received codeword must be {n*8} bits for RS({n},{k}).")

    rec_symbols = np.packbits(received_bits).astype(np.uint8)
    reliability = compute_symbol_reliability(segment_bits, n=n, k=k)

    decoded_bytes = rs_osd1_decode(
        rs=rs,
        received_symbols=rec_symbols,
        reliability_symbols=reliability,
        segment_stats=segment_stats,
        K=K,
        L=L
    )

    if decoded_bytes is None:
        return received_bits[:k * 8]  # fail-safe

    return np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8))
 
# ================================================================
# DEMO
# ================================================================
if __name__ == "__main__":

    # ----------- 1. Generate a random 24-bit message --------------
    rng = np.random.default_rng(1)
    msg_bits = rng.integers(0, 2, size=24, dtype=np.uint8)
    print("Message bits (24):", msg_bits)

    # ----------- 2. Encode to 48-bit RS codeword ------------------
    code_bits = rs_encode_bits(msg_bits)
    print("Encoded bits (40):", code_bits)

    # ----------- 3. Inject BIT errors -----------------------------
    noisy = code_bits.copy()
    flip_positions = [16, 17, 18]      # try staying inside one symbol region
    for p in flip_positions:
        noisy[p] ^= 1

    print("Received bits:", noisy)

    # ----------- 4. Decode via OSD-1 + RS ------------------------
    decoded_bits = rs_decode_bits(noisy, segment_bits)
    print("Decoded bits:", decoded_bits)

    print("Correct decode? ", np.array_equal(decoded_bits, msg_bits))

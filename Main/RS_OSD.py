import numpy as np
import reedsolo
from itertools import combinations

# ================================================================
# RS PARAMETERS
# ================================================================
n = 5      # RS symbols (bytes)
k = 3      # message symbols
rs = reedsolo.RSCodec(n - k)
t = (n - k) // 2   # = 1 symbol correctable


# ================================================================
# SEGMENT-TO-BIT MAPPING (your example)
# ================================================================


# For demonstration, we map segments to bit positions in a
# simple sequential way:
def get_segment_reliability(bit_length):
    """Linear reliability = 1 / (bit-length)."""
    return 1.0 / bit_length


def compute_symbol_reliability(segment_bits):
    """
    Convert segment-based bit reliabilities into RS symbol-level reliabilities.
    We assume the 24 message bits map sequentially into 3 bytes (3*8 bits).
    """
    TOTAL_BITS = 40       # RS(6,3) codeword = 6 bytes = 48 bits
    BITS_PER_SYMBOL = 8
    NUM_SYMBOLS = 5

    reliability_bits = np.zeros(TOTAL_BITS, dtype=float)

    bit_pos = 0
    for seg, bits in segment_bits.items():
        L = len(bits)              # segment length
        rel = 1.0 / L              # reliability per bit
        for i in range(L):
            if bit_pos < TOTAL_BITS:
                reliability_bits[bit_pos] = rel
            bit_pos += 1

    # if fewer than 48 bits assigned, fill rest with reliability 1.0
    if bit_pos < TOTAL_BITS:
        reliability_bits[bit_pos:TOTAL_BITS] = 1.0

    # ---- Convert bit reliabilities → symbol reliabilities ----
    symbol_rel = np.zeros(NUM_SYMBOLS)
    for i in range(NUM_SYMBOLS):
        start = i * BITS_PER_SYMBOL
        end   = (i+1) * BITS_PER_SYMBOL
        symbol_rel[i] = np.sum(reliability_bits[start:end])  # SUM, not mean

    return symbol_rel


# ================================================================
# BIT ↔ SYMBOL UTILITIES
# ================================================================
def bits_to_bytes(bits24):
    """Convert 24 bits → 3 bytes."""
    bits24 = np.array(bits24, dtype=np.uint8)
    if bits24.size != 24:
        raise ValueError("Message must be exactly 24 bits.")
    return np.packbits(bits24).astype(np.uint8)


def bytes_to_bits(bts):
    """Convert bytes → bits array."""
    return np.unpackbits(np.frombuffer(bts, dtype=np.uint8))


# ================================================================
# ENCODING
# ================================================================
def rs_encode_bits(bits24):
    """
    Input: 24 bits (numpy array of 0/1)
    Output: 48 bits (encoded RS(6,3) codeword)
    """
    msg_bytes = bits_to_bytes(bits24)   # 3 bytes
    encoded = rs.encode(msg_bytes.tobytes())  # 6 bytes
    return np.unpackbits(np.frombuffer(encoded, dtype=np.uint8))


# ================================================================
# OSD-1 + REED-SOLOMON DECODER
# ================================================================
def rs_osd1_decode(rs, received_symbols, reliability_symbols):
    """
    OSD-1 (Chase-1) soft-decision decoding for RS.
    """
    n = len(received_symbols)
    least_rel = np.argsort(reliability_symbols)

    # Order-0 candidate
    candidates = [received_symbols]

    # OSD-1 flips for L least reliable positions
    L = 2
    for pos in least_rel[:L]:
        cand = received_symbols.copy()
        cand[pos] ^= 0xFF   # flip all bits in symbol
        candidates.append(cand)

    # Try all candidates
    best_msg = None
    best_metric = 9e99

    for cand in candidates:
        try:
            decoded = rs.decode(bytes(cand))[0]  # message bytes
        except reedsolo.ReedSolomonError:
            continue

        # soft metric: weighted differences
        dist = 0
        for i in range(n):
            if cand[i] != received_symbols[i]:
                dist += 1.0 / (reliability_symbols[i] + 1e-9)

        if dist < best_metric:
            best_metric = dist
            best_msg = decoded

    return best_msg


# ================================================================
# DECODING FROM RECEIVED BITS
# ================================================================
def rs_decode_bits(received_bits48, segment_bits):
    """
    Input: 48 bits (received)
    Output: 24 bits (decoded message)
    """
    rec_bits = np.array(received_bits48, dtype=np.uint8)
    #print(rec_bits)
    if rec_bits.size != 40:
        raise ValueError("Received codeword must be 48 bits.")

    # Convert to 6 symbols
    rec_symbols = np.packbits(rec_bits).astype(np.uint8)

    # Compute symbol-level reliability from segment definitions
    reliability = compute_symbol_reliability(segment_bits)

    # OSD + RS decode
    decoded_bytes = rs_osd1_decode(rs, rec_symbols, reliability)
    if decoded_bytes is None:
        fallback_bits = rec_bits[:24]  # or whatever message length is
        return fallback_bits.astype(np.uint8)

    # Convert to bits
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

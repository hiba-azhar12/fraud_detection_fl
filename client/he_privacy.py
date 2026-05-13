import numpy as np
import os

try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except ImportError:
    TENSEAL_AVAILABLE = False
    print("[HE] TenSEAL non disponible — mode simulation active.")


class HEContext:

    def __init__(self):
        if not TENSEAL_AVAILABLE:
            self.ctx = None
            return

        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree = 8192,
            coeff_mod_bit_sizes = [60, 40, 40, 60],
        )
        self.ctx.global_scale = 2 ** 40
        self.ctx.generate_galois_keys()

    def serialize(self) -> bytes:
        if self.ctx is None:
            return b""
        return self.ctx.serialize(save_secret_key=False)

    @staticmethod
    def from_bytes(data: bytes) -> "HEContext":
        obj = HEContext.__new__(HEContext)
        obj.ctx = ts.context_from(data) if TENSEAL_AVAILABLE else None
        return obj


def encrypt_gradients(grad_list: list, he_ctx: HEContext) -> list:
    if not TENSEAL_AVAILABLE or he_ctx.ctx is None:
        return [("SIMULATED", g.shape, g.dtype, g) for g in grad_list]

    encrypted = []
    for g in grad_list:
        flat      = g.astype(np.float64).flatten().tolist()
        enc_vec   = ts.ckks_vector(he_ctx.ctx, flat)
        enc_bytes = enc_vec.serialize()
        encrypted.append((enc_bytes, g.shape, g.dtype))
    return encrypted


def decrypt_gradients(encrypted_list: list, he_ctx: HEContext) -> list:
    if not TENSEAL_AVAILABLE or he_ctx.ctx is None:
        return [item[3] for item in encrypted_list]

    result = []
    for enc_bytes, shape, dtype in encrypted_list:
        enc_vec  = ts.ckks_vector_from(he_ctx.ctx, enc_bytes)
        dec_flat = np.array(enc_vec.decrypt())
        result.append(dec_flat.reshape(shape).astype(dtype))
    return result


def aggregate_encrypted(encrypted_lists: list, he_ctx: HEContext) -> list:
    if not TENSEAL_AVAILABLE:
        result = []
        for layers in zip(*encrypted_lists):
            stacked = np.stack([item[3] for item in layers])
            result.append(("SIMULATED", layers[0][1], layers[0][2], stacked.mean(axis=0)))
        return result

    n_layers   = len(encrypted_lists[0])
    aggregated = []

    for layer_idx in range(n_layers):
        enc_bytes_0, shape, dtype = encrypted_lists[0][layer_idx]
        acc = ts.ckks_vector_from(he_ctx.ctx, enc_bytes_0)

        for client_enc in encrypted_lists[1:]:
            enc_bytes_i, _, _ = client_enc[layer_idx]
            vec_i = ts.ckks_vector_from(he_ctx.ctx, enc_bytes_i)
            acc   = acc + vec_i

        acc = acc * (1.0 / len(encrypted_lists))
        aggregated.append((acc.serialize(), shape, dtype))

    return aggregated

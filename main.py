
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.sparse.linalg import LinearOperator, cg
from PIL import Image
import time
import os
from tqdm.auto import tqdm

def dct2(x):
    # apply 1D dct (type II, orthonormal) on rows then cols
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    # inverse DCT (type III, orthonormal)
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def psnr(x_hat, x_true):
    N = x_true.size
    num = np.max(np.abs(x_true))
    den = np.linalg.norm((x_true - x_hat).ravel()) / np.sqrt(N)
    if den == 0:
        return np.inf
    return 20.0 * np.log10(num / den)

def rel_l2(x_hat, x_true):
    return np.linalg.norm(x_hat.ravel() - x_true.ravel()) / np.linalg.norm(x_true.ravel())

def make_random_mask(shape, sampling_ratio, rng=None):
    rng = np.random.default_rng(rng)
    N = shape[0] * shape[1]
    M = int(np.round(sampling_ratio * N))
    idx = rng.choice(N, size=M, replace=False)
    mask = np.zeros(N, dtype=np.uint8)
    mask[idx] = 1
    return mask.reshape(shape)

def apply_mask(x, mask):
    return mask * x

def mm_cg_reconstruct(m, mask, lam, p=0.4, eps=1e-6,
                      tol_cg=1e-6, tol_mm=1e-4, max_mm_iters=200,
                      verbose=False):
    shape = m.shape
    N = shape[0] * shape[1]
    # initialize zero-filled image
    xk = m.copy()
    Wt_m = m.copy()  # because W^T m is just mask * m (measurements in place)
    diag_mask = mask  # M operator is elementwise multiply by mask

    obj_vals = []
    rel_changes = []
    cg_iters_per_mm = []
    times = []

    for k in tqdm(range(max_mm_iters)):
        t0 = time.time()
        # DCT of current estimate
        y_hat = dct2(xk)
        # compute weights in DCT domain: w_i = p * (eps + y_i^2)^{p-1}
        w = p * (eps + y_hat**2)**(p - 1)

        # define linear operator M^{(k)} acting on image-shaped vectors
        def M_op(vec_flat):
            z = vec_flat.reshape(shape)
            term1 = diag_mask * z
            # DCT domain multiplication then IDCT
            Dz = idct2(w * dct2(z))
            return (term1 + lam * Dz).ravel()

        linop = LinearOperator((N, N), matvec=M_op, dtype=np.float64)

        # right-hand side is W^T m = mask * m
        b = Wt_m.ravel()

        # CG solve: M^{(k)} x = b
        # use previous xk as initial guess
        x0 = xk.ravel()
        x_sol, info = cg(linop, b, x0=x0, tol=tol_cg, maxiter=10*N)
        # info==0 : converged; >0 number of iterations, <0 error

        x_next = x_sol.reshape(shape)

        # diagnostics
        # objective value J(x)
        data_fidelity = np.linalg.norm(apply_mask(x_next, mask) - m)**2
        y_next = dct2(x_next)
        reg = lam * np.sum((eps + y_next**2)**p)
        Jk = data_fidelity + reg
        obj_vals.append(Jk)

        rel_change = np.linalg.norm((x_next - xk)) / (np.linalg.norm(xk) + 1e-12)
        rel_changes.append(rel_change)
        cg_iters_per_mm.append(info if isinstance(info, (int, np.integer)) else 0)
        times.append(time.time() - t0)

        if verbose:
            print(f"MM iter {k:3d}: J={Jk:.6e}, rel_change={rel_change:.3e}, cg_info={info}")

        # stopping
        xk = x_next
        if rel_change < tol_mm:
            if verbose:
                print("MM stopping criterion reached.")
            break

    diagnostics = {
        'obj_vals': np.array(obj_vals),
        'rel_changes': np.array(rel_changes),
        'cg_iters': np.array(cg_iters_per_mm),
        'times': np.array(times),
        'iters': k + 1
    }
    return xk, diagnostics
def run_experiments(image_paths, sampling_ratios=[0.1,0.2,0.3,0.5],
                    p_list=[0.3,0.4,0.5],
                    lambda_grid=[0.001, 0.001, 0.01, 0.1, 1],
                    snr_db=30.0,
                    eps=1e-6,
                    rng_seed=0,
                    out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    all_results = []

    for img_path in image_paths:
        # load and normalize to [0,1]
        im = Image.open(img_path).convert('L')
        im = im.resize((256,256))
        x_true = np.asarray(im, dtype=np.float32) / 255.0

        for r in sampling_ratios:
            mask = make_random_mask(x_true.shape, r, rng=rng)

            sampled = mask * x_true
            observed_energy = np.linalg.norm(sampled)**2
            eta_norm = np.linalg.norm(sampled) / (10**(snr_db / 20.0) + 1e-12)

            eta = np.zeros_like(x_true)
            noise_random = rng.standard_normal(size=x_true.shape)
            noise_random *= mask
            if np.linalg.norm(noise_random) == 0:
                noise_random = mask * 1e-12
            noise_random *= (eta_norm / np.linalg.norm(noise_random))
            eta = noise_random
            m = sampled + eta

            # validation sweep: try all (p, lambda) combinations, keep best PSNR
            best = {'psnr': -np.inf}
            results_table = []
            
            for p in p_list:
                for lam in lambda_grid:
                    x_hat, diag = mm_cg_reconstruct(
                        m, mask, lam, p=p, eps=eps,
                        tol_cg=1e-6, tol_mm=1e-4,
                        max_mm_iters=200, verbose=False
                    )
                    cur_psnr = psnr(x_hat, x_true)
                    cur_rel = rel_l2(x_hat, x_true)
                    results_table.append((p, lam, cur_psnr, cur_rel, diag))
                    all_results.append((f"{img_path}", r, p, lam, cur_psnr))
                    if cur_psnr > best['psnr']:
                        best = {
                            'p': p, 'lam': lam,
                            'psnr': cur_psnr, 'rel': cur_rel,
                            'x_hat': x_hat, 'diag': diag
                        }

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_prefix = f"{base_name}_r{int(r*100)}"

            plt.imsave(os.path.join(out_dir, f"{out_prefix}_mask.png"),  mask,          cmap='gray')
            plt.imsave(os.path.join(out_dir, f"{out_prefix}_noisy.png"), np.clip(m,0,1), cmap='gray')
            plt.imsave(
                os.path.join(out_dir, f"{out_prefix}_recon_p{best['p']}_lam{best['lam']}.png"),
                np.clip(best['x_hat'],0,1),
                cmap='gray'
            )

            # PSNR vs lambda plot (for each p)
            plt.figure(figsize=(6,4))
            for p in p_list:
                lambdas = []
                psnrs = []
                for (pp, lam, ps, rl, dg) in results_table:
                    if pp == p:
                        lambdas.append(lam)
                        psnrs.append(ps)
                lambdas = np.array(lambdas)
                psnrs = np.array(psnrs)
                order = np.argsort(lambdas)
                plt.plot(lambdas[order], psnrs[order], marker='o', label=f"p={p}")
            plt.xscale('log')
            plt.xlabel('lambda (log scale)')
            plt.ylabel('PSNR (dB)')
            plt.title(f"{base_name} r={r}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"{out_prefix}_psnr_vs_lambda.png"))
            plt.close()

            # convergence curves for best setting
            diag = best['diag']
            plt.figure(figsize=(6,4))
            plt.plot(diag['obj_vals'])
            plt.xlabel('MM iteration')
            plt.ylabel('Objective J')
            plt.title(f"Convergence (p={best['p']}, lam={best['lam']})")
            plt.grid(True)
            plt.savefig(os.path.join(out_dir, f"{out_prefix}_convergence.png"))
            plt.close()

            print(
                f"{base_name} r={r}: best p={best['p']}, lam={best['lam']}, "
                f"PSNR={best['psnr']:.2f} dB, rel={best['rel']:.4f}"
            )

    print("All experiments finished. Results saved in:", out_dir)
    return all_results



if __name__ == "__main__":
    # update paths to your images (cameraman, barbara/lena)
    image_paths = ["cameraman.tif","lena.png"]  # ensure these files exist in working dir
    res = run_experiments(image_paths,
                    sampling_ratios=[0.1,0.2,0.3,0.5],
                    p_list=[0.3,0.4,0.5],
                    lambda_grid=[1e-4,1e-3,1e-2,1e-1,1],
                    snr_db=30.0,
                    eps=1e-6,
                    rng_seed=42,
                    out_dir='results_mm_cg')
    print(res)

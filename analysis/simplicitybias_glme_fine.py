# ============================================================
# Hierarchical power-law correction model in NumPyro (NUTS / MCMC)
# Data columns expected in df_alltrials:
#  - Subject_global (int, arbitrary IDs across experiments)
#  - response_choosecomplexmodel (0/1)
#  - OLSnoise_est_mean (float > 0)     # or swap to noise_var if you prefer
#  - sample_size_inv  (float > 0)      # 1/N_train
#  - dD  (ΔD = D_simple - D_complex; in your data ΔD < 0)
#  - trainMSE_modeldiff (ΔtrainMSE = trainMSE_simple - trainMSE_complex)
#     NOTE: More positive ΔtrainMSE favors choosing the complex model.
# ============================================================
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pickle
import os

def randomSlopes_NUTS(num_warmup=1000, num_samples=1000, num_chains=2, rng_seed=0):
    
    print("num_warmup={0}, num_samples={1}, num_chains={2}, seed={3}".format(num_warmup, num_samples, num_chains, rng_seed))

    # Set the number of CPU devices for parallel chains (e.g., 4)
    numpyro.set_host_device_count(num_chains) 
    import jax # Import JAX after setting host device count
    num_devices_after_config = jax.local_device_count()
    print("Number of JAX devices after configuration:", str(num_devices_after_config))

    # -----------------------
    # 5) Putting it together
    # -----------------------
    df_alltrials = pd.read_csv(os.path.join("simplicity_bias", "df_alltrials.csv"))
    data = prepare_tensors(df_alltrials)
    mcmc = fit_nuts_np(data, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                       rng_seed=rng_seed, chain_method="parallel")
    # Collect the results
    posterior_samples = mcmc.get_samples()
    _ = summarize_posterior_mcmc_np(mcmc)
    pp_mean, pp_draws = posterior_predictive_probs_np(mcmc, data)
    acc = np.mean(((pp_mean > 0.5).astype(np.float32) == np.array(data["y"])))
    print("Posterior predictive mean accuracy:", float(acc))

    file_path = os.path.join("simplicity_bias", "data","glme_numpyro_"+str(num_warmup)+"_"+str(num_samples)+"_fine_seed"+str(rng_seed)+".pickle")
    a = {"data":data, "mcmc":mcmc, "posterior_samples":posterior_samples, "pp_mean":pp_mean, "pp_draws":pp_draws, "acc": acc, "num_warmup": num_warmup, "num_samples": num_samples, "num_chains": num_chains, "rng_seed": rng_seed}
    with open(file_path, 'wb') as file_handle:
        pickle.dump(a, file_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Data files saved.")
    mcmc.print_summary()
    

    
# -----------------------
# 0) Prepare tensors (-> JAX DeviceArrays)
# -----------------------
def prepare_tensors(df):
    subj_codes = pd.Categorical(df["Subject_global"])
    subj_idx = jnp.array(subj_codes.codes, dtype=jnp.int32)

    y = 1-jnp.array(df["response_choosecomplexmodel"].values, dtype=jnp.float32)

    # V = jnp.array(df["OLSnoise_est_mean"].values, dtype=jnp.float32)
    V = jnp.array(df["noise_var"].values, dtype=jnp.float32)

    invN   = jnp.array(df["sample_size_inv"].values, dtype=jnp.float32)
    dD_raw = -jnp.array(df["dD"].values, dtype=jnp.float32)
    dTrain = -jnp.array(df["trainMSE_modeldiff"].values, dtype=jnp.float32)

    dTrain_z = dTrain  # keep unscaled to match your Pyro code

    S = int(subj_codes.categories.size)
    N = len(df)

    eps = 1e-12
    return {
        "y": y,
        "V": jnp.clip(V, a_min=eps),
        "invN": jnp.clip(invN, a_min=eps),
        "dD_raw": dD_raw,          # keep signed; do NOT clip here
        "dTrain_z": dTrain_z,
        "subj_idx": subj_idx,
        "S": S,
        "N": N,
    }

# -----------------------
# 1) Model (with signed power for ΔD)
# -----------------------
def powerlaw_logit_model_fine(y, V, invN, dD_raw, dTrain_z, subj_idx, S):
    """
    logit P(y=1) = θ0_s + θtrain_s * dTrain_z  - κ_s * (V^α) * [sign(ΔD)*|ΔD|^β] * (invN^γ)

    - Signed power ensures well-defined ΔD^β when ΔD<0 and β∈ℝ.
    - α, β, γ: population exponents (shared).
    - θ0_s, θtrain_s, κ_s: subject-level via non-centered parameterization; κ_s unconstrained.
    """

    # ---- Population exponents (normative target = 1) ----
    alpha = numpyro.sample("alpha", dist.Normal(1, 1))
    beta  = numpyro.sample("beta",  dist.Normal(1, 1))
    gamma = numpyro.sample("gamma", dist.Normal(1, 1))

    # ---- Hyperpriors for subject-level effects ----
    # Intercepts
    mu0    = numpyro.sample("mu0",    dist.Normal(0, 10))
    sigma0 = numpyro.sample("sigma0", dist.HalfCauchy(2.5))
    # Slopes on ΔtrainMSE
    mu_t    = numpyro.sample("mu_t",    dist.Normal(1, 10))
    sigma_t = numpyro.sample("sigma_t", dist.HalfCauchy(2.5))
    # κ hierarchy (unconstrained, as in your Pyro code)
    mu_k   = numpyro.sample("mu_k",  dist.Normal(1, 10))
    sig_k  = numpyro.sample("sig_k", dist.HalfCauchy(2.5))

    with numpyro.plate("subjects", S):
        z0 = numpyro.sample("z0", dist.Normal(0.0, 1))
        zt = numpyro.sample("zt", dist.Normal(0.0, 1))
        zk = numpyro.sample("zk", dist.Normal(0.0, 1))

        theta0_s     = numpyro.deterministic("theta0_s",     mu0  + sigma0  * z0)
        thetatrain_s = numpyro.deterministic("thetatrain_s", mu_t + sigma_t * zt)
        kappa_s      = numpyro.deterministic("kappa_s",      mu_k + sig_k   * zk)  # could be ±

    ## Signed power for ΔD
    # sign_dD = jnp.sign(dD_raw)
    # abs_dD  = jnp.clip(jnp.abs(dD_raw), a_min=1e-12)

    ## Power-law correction term
    # corr = (V ** alpha) * (sign_dD * (abs_dD ** beta)) * (invN ** gamma)  # [N]
    corr = (V ** alpha) * (dD_raw ** beta) * (invN ** gamma)  # [N]

    # Gather subject-level params per trial
    theta0_per     = theta0_s[subj_idx]      # [N]
    thetatrain_per = thetatrain_s[subj_idx]  # [N]
    kappa_per      = kappa_s[subj_idx]       # [N]

    # Linear predictor (minus sign encodes normative direction)
    eta = theta0_per + thetatrain_per * dTrain_z + 2*kappa_per * corr

    with numpyro.plate("data", y.shape[0]):
        numpyro.sample("y_obs", dist.Bernoulli(logits=eta), obs=y)

# -----------------------
# 2) Fit with NUTS / MCMC
# -----------------------
def fit_nuts_np(data, num_warmup=1000, num_samples=1000, num_chains=1,
                rng_seed=0, progress_bar=True, chain_method="sequential",
                target_accept_prob=0.9, max_tree_depth=12):
    """
    chain_method: "sequential" (safe everywhere) or "parallel" (requires XLA multi-processing support).
    """
    numpyro.set_host_device_count(num_chains)
    kernel = NUTS(powerlaw_logit_model_fine,
                  target_accept_prob=target_accept_prob,
                  max_tree_depth=max_tree_depth)

    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar,
                chain_method=chain_method)

    rng_key = jax.random.PRNGKey(rng_seed)
    mcmc.run(rng_key,
             data["y"], data["V"], data["invN"], data["dD_raw"],
             data["dTrain_z"], data["subj_idx"], data["S"])
    return mcmc

# -----------------------
# 3) Posterior summaries
# -----------------------
def summarize_posterior_mcmc_np(mcmc):
    s = mcmc.get_samples(group_by_chain=False)  # dict of [draws] or [draws, S]

    def ci(a, q=(0.025, 0.5, 0.975)):
        qs = jnp.quantile(a, jnp.array(q))
        return {"lo": float(qs[0]), "med": float(qs[1]), "hi": float(qs[2])}

    print("\nPosterior (median [95% CI])")
    for name in ["alpha","beta","gamma","mu0","sigma0","mu_t","sigma_t","mu_k","sig_k"]:
        stats = ci(s[name])
        print(f"{name:>8s}: {stats['med']:.3f}  [{stats['lo']:.3f}, {stats['hi']:.3f}]")

    for name in ["alpha","beta","gamma"]:
        p_above = float(jnp.mean((s[name] - 1.0) > 0.0))
        print(f"P({name} > 1) = {p_above:.3f} ;  E[{name}] ≈ {float(jnp.mean(s[name])):.3f}")

    return s

# -----------------------
# 4) Posterior predictive probabilities (per trial)
# -----------------------
def posterior_predictive_probs_np(mcmc, data, thin=None):
    """
    Returns:
      probs_mean: [N] posterior predictive mean P(y=1) per trial
      probs_all:  [draws, N] raw per-draw probabilities (for CIs)
    """
    s = mcmc.get_samples(group_by_chain=False)

    # Optional thinning
    if thin is not None and thin > 1:
        idx = jnp.arange(0, s["alpha"].shape[0], step=thin, dtype=jnp.int32)
        s = {k: v[idx] for k, v in s.items()}

    # Reconstruct subject-level params
    theta0 = s["mu0"][:, None]  + s["sigma0"][:, None]  * s["z0"]   # [draws, S]
    thetat = s["mu_t"][:, None] + s["sigma_t"][:, None] * s["zt"]   # [draws, S]
    kappa  = s["mu_k"][:, None] + s["sig_k"][:, None]   * s["zk"]   # [draws, S]

    # Broadcast scalars to [draws, N]
    alpha = s["alpha"][:, None]
    beta  = s["beta"][:, None]
    gamma = s["gamma"][:, None]

    V     = data["V"][None, :]        # [1, N]
    invN  = data["invN"][None, :]     # [1, N]
    dD    = data["dD_raw"][None, :]   # [1, N]
    dTr   = data["dTrain_z"][None, :] # [1, N]
    subj  = data["subj_idx"]          # [N]

    # sign_dD = jnp.sign(dD)
    # abs_dD  = jnp.clip(jnp.abs(dD), a_min=1e-12)

    # corr = (V ** alpha) * (sign_dD * (abs_dD ** beta)) * (invN ** gamma)  # [draws, N]
    corr = (V ** alpha) * (dD ** beta) * (invN ** gamma)  # [draws, N]

    # Gather subject-level params per trial (index 2nd axis by subj)
    theta0_per = jnp.take(theta0, subj, axis=1)  # [draws, N]
    thetat_per = jnp.take(thetat, subj, axis=1)  # [draws, N]
    kappa_per  = jnp.take(kappa,  subj, axis=1)  # [draws, N]

    eta = theta0_per + thetat_per * dTr + 2*kappa_per * corr
    probs_all = jax.nn.sigmoid(eta)              # [draws, N]
    probs_mean = jnp.mean(probs_all, axis=0)     # [N]
    return np.array(probs_mean), np.array(probs_all)




randomSlopes_NUTS(num_warmup=2000, num_samples=2000, num_chains=4, rng_seed=2)
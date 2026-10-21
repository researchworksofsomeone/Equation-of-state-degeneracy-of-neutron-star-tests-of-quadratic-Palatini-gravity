# Palatini gravity: neutron-star degeneracy, a muonic-atom bound, and the infrared sector

Code accompanying the paper *"Equation-of-state degeneracy of neutron-star tests of quadratic Palatini gravity and a lepton-universality bound from electronic and muonic charge radii."*

| What | Where |
|---|---|
| Notebook (canonical, reactive): `palatini_jeans_window.py` | this repository |
| Same notebook for Jupyter: `palatini_jeans_window.ipynb` | this repository |
| All checkpoints, JSON exports of every quoted number, paper figures | [Hugging Face dataset `palatini-molab-runs`](https://huggingface.co/datasets/APassionateResearchers/palatini-molab-runs/tree/main) |
| Release builder (strips secrets, exports `.ipynb`) | `make_release.py` |

The notebook was written for [marimo](https://marimo.io) and run on molab (one GPU with 70 GB VRAM, 30 GB RAM). Every expensive stage writes per-point checkpoints, so any stage can be resumed, and the whole paper can be regenerated from the published checkpoints without a GPU.

---

## Contents

1. [Quick start](#1-quick-start)
2. [Reproducing the paper from the published checkpoints](#2-reproducing-the-paper-from-the-published-checkpoints)
3. [Notebook map](#3-notebook-map)
4. [Mathematics](#4-mathematics)
5. [Data](#5-data)
6. [Numerical methods and validation](#6-numerical-methods-and-validation)
7. [Results reproduced by the notebook](#7-results-reproduced-by-the-notebook)
8. [Pre-registered tests and referee checks](#8-pre-registered-tests-and-referee-checks)
9. [Checkpoint layout on Hugging Face](#9-checkpoint-layout-on-hugging-face)
10. [Environment variables](#10-environment-variables)
11. [Security and anonymity](#11-security-and-anonymity)

---

## 1. Quick start

```bash
pip install -r requirements.txt          # marimo numpy scipy matplotlib psutil camb torch huggingface_hub
marimo edit palatini_jeans_window.py     # or open it on molab
```

* `SMOKE=1 marimo edit palatini_jeans_window.py` runs every stage at a tiny size (minutes on a CPU) to check the pipeline.
* Heavy stages start only when their button is pressed, or headlessly with `RUN_HEAVY=1`, `RUN_NS=1`, `RUN_P3=1`, `RUN_P4=1`, `RUN_P7=1` (Section 10).
* In Jupyter (`palatini_jeans_window.ipynb`), run the cells from top to bottom. Buttons do not fire in Jupyter, so either set the `RUN_*` variables before starting the kernel or restore the checkpoints first (Section 2), in which case every stage loads instead of recomputing.

## 2. Reproducing the paper from the published checkpoints

Checkpoints are available at [https://huggingface.co/datasets/APassionateResearchers/palatini-molab-runs/tree/main](https://huggingface.co/datasets/APassionateResearchers/palatini-molab-runs/tree/main).

```python
from huggingface_hub import snapshot_download
snapshot_download("APassionateResearchers/palatini-molab-runs", repo_type="dataset",
                  allow_patterns=["latest/pj_ckpt/*"], local_dir="hf")
# then move hf/latest/pj_ckpt -> ./pj_ckpt and run the notebook; finished points are skipped
```

The figure cell (cell 72) regenerates every paper figure from `pj_ckpt/` alone; cell 74 copies them to `figures/` under the names used in the LaTeX source (`fig1_bound.png`, `fig2_stars.png`, `figS1_cosmology.png` … `figS6_radii.png`). The ready figures are also in `paper/figures/` on the dataset. Two supplementary figures (`figS5_posterior`, `figS6_radii`) had their legends moved off the data in the final notebook, so regenerating them gives a slightly cleaner layout than the copies on the dataset; the plotted data are identical.

## 3. Notebook map

| Part | Cells | What it computes | Main export |
|---|---|---|---|
| Setup | 1–3 | environment, memory budgets, configuration | — |
| I | 4–21 | infrared Lagrangian $f=R-2\Lambda-\mu^4R^{-n}$: background, quasi-static growth, halofit, lensing, joint cosmological fit (profile likelihood + MCMC), robustness | `pj_ckpt/RESULTS_<tag>.json`, `PROFILE_<tag>.json`, `mcmc_*.npz` |
| II | 22–31 | infrared sector inside neutron stars: exact trace map, conformal wall, batched TOV + Love numbers, evidences | `part2/PART2_RESULTS.json` |
| III | 32–40 | Einstein-frame theorem, pQCD constraint, optional sample-based likelihoods, prior variants | `part3/PART3_RESULTS.json` |
| IV–V | 41–49 | Gaussian-process EOS prior, degeneracy, discovery space | `part4/PART4_RESULTS.json`, `PART5_FORECAST.json` |
| VI | 50–54 | compactness bound for causal, nowhere-superconformal matter (GR) | `part4/PART6_COMPACTNESS.json` |
| VII | 55–62 | full quadratic Lagrangian $R+\alpha R^2+\beta R_{(\mu\nu)}R^{(\mu\nu)}$: exact map, crust condition, crustal inertia, 20 000 EOS × 46 couplings | `part7/a*_b*_batch*.npz`, `part7/PART7_RBG.json` |
| VIII | 63–69 | contact interaction, frame invariance, lepton-universality bound, unified crusts, robustness, claim ledger | `part4/PART8_*.json` |
| Figures | 72, 74 | all paper figures; copy to `figures/` and upload | `paper_figs/`, `figures/` |
| Backup | 73 | sync checkpoints to Hugging Face (token read from the environment) | — |
| Referee checks | 75–79 (the last five cells) | atomic robustness, closed-form checks, Monte Carlo and prior sensitivity, sample-based likelihoods | `referee/*.json`, `part7_kde/` |

## 4. Mathematics

Conventions: signature $(-,+,+,+)$; mixed tensors are $4\times4$ matrices; $\mathrm{tr}$ is the matrix trace. Stellar structure uses $G=c=1$, lengths in km, $\kappa=8\pi$, couplings $\alpha,\beta$ in km². Atomic physics uses $\hbar=c=1$, $\kappa=8\pi/M_{\rm Pl}^2$; one km² is $2.568\times10^{37}\,\mathrm{GeV^{-2}}$ and $\kappa\times(1\,\mathrm{km^2})=4.33\,\mathrm{GeV^{-4}}$.

### 4.1 Action and field equations

$$
S=\frac{1}{2\kappa}\int d^4x\,\sqrt{-g}\,f(R,Q)+S_m[g,\psi],\qquad R=g^{\mu\nu}R_{(\mu\nu)}(\Gamma),\quad Q=R_{(\mu\nu)}R^{(\mu\nu)},
$$

with an independent torsion-free connection $\Gamma$ and matter minimally coupled to $g$. Parts VII–VIII use $f=R+\alpha R^2+\beta Q$; Parts I–VI use $f=R-2\Lambda-\mu^4R^{-n}$.

With $P^\mu{}_\nu=g^{\mu\alpha}R_{(\alpha\nu)}$, the metric equations are the matrix equation

$$
\Omega P-\tfrac f2\,\mathbf 1=\kappa T,\qquad \Omega=f_R\mathbf 1+2\beta P .
$$

The connection equation is $\nabla^\Gamma_\lambda(\sqrt{-g}\,g^{\mu\alpha}\Omega_\alpha{}^\nu)=0$, so $\Gamma$ is the Levi-Civita connection of

$$
q_{\mu\nu}=|\Omega|^{1/2}g_{\mu\alpha}(\Omega^{-1})^\alpha{}_\nu,\qquad \sqrt{-q}=|\Omega|^{1/2}\sqrt{-g},
$$

and $q$ obeys Einstein's equations $G^\mu{}_\nu(q)=\kappa\tilde T^\mu{}_\nu$ with

$$
\tilde R^\mu{}_\nu=|\Omega|^{-1/2}\big(\kappa T^\mu{}_\nu+\tfrac f2\delta^\mu_\nu\big),\qquad \kappa\tilde T^\mu{}_\nu=\tilde R^\mu{}_\nu-\tfrac12\tilde R\,\delta^\mu_\nu .
$$

**Trace identity.** For the quadratic Lagrangian the trace gives $R=-\kappa T$ exactly, for all $\alpha,\beta$.

**Algebraic structure.** The solution connected to GR, $P_{\rm GR}=\kappa(T-\tfrac12T\mathbf 1)$, is a polynomial in $T$ (Cayley–Hamilton plus the implicit-function theorem), so $\tilde T$ shares the eigenvectors of $T$.

### 4.2 Perfect fluids: the effective equation of state

For $T={\rm diag}(-\varepsilon,p,p,p)$, $P={\rm diag}(P_0,P_1,P_1,P_1)$ solves

$$
2\beta P_a^2+(1+2\alpha R)P_a-\tfrac12\big(R+\alpha R^2+\beta(P_0^2+3P_1^2)\big)=\kappa T_a,\qquad T_0=-\varepsilon,\ T_1=p,
$$

and with $\sigma_a=1+2\alpha R+2\beta P_a$, $\tilde R_a=\sigma_aP_a/\sqrt{\sigma_0\sigma_1^3}$,

$$
\kappa\tilde\varepsilon=\tfrac12(3\tilde R_1-\tilde R_0),\qquad \kappa\tilde p=-\tfrac12(\tilde R_0+\tilde R_1).
$$

A Palatini star is therefore a GR star of the effective fluid $(\tilde\varepsilon,\tilde p)$. For $\beta=0$, $F=1+2\alpha R$ and

$$
\tilde\varepsilon=\frac{\varepsilon}{F^2}+\frac{\alpha R^2}{2\kappa F^2},\qquad \tilde p=\frac{p}{F^2}-\frac{\alpha R^2}{2\kappa F^2}.
$$

**Conformal selection rule.** For $\beta=0$, $\tilde T=T/F$ exactly: the $R^2$ term never changes the sign of the trace anomaly $\Delta=\tfrac13-p/\varepsilon$ and leaves conformal matter unchanged.

**First order in the couplings:**

$$
\delta\tilde\varepsilon=-\tfrac{\alpha\kappa}{2}(7\varepsilon+3p)(\varepsilon-3p)-\tfrac{\beta\kappa}{2}(\varepsilon^2-6\varepsilon p-3p^2),\qquad
\delta\tilde p=-\tfrac{\alpha\kappa}{2}(\varepsilon+5p)(\varepsilon-3p)-\tfrac{\beta\kappa}{2}(\varepsilon+3p)(\varepsilon-p).
$$

**Second order, $\beta=0$:** $\delta^{(2)}\tilde\varepsilon=\alpha^2\kappa^2(\varepsilon-3p)^2(10\varepsilon+6p)$ and $\delta^{(2)}\tilde p=\alpha^2\kappa^2(\varepsilon-3p)^2(2\varepsilon+6p)$.

**Exact effective sound speed, $\beta=0$** ($a=\alpha\kappa$, $D=\varepsilon-3p$):

$$
\tilde c_s^2=\frac{c_s^2(1+2aD)-a(1-3c_s^2)(\varepsilon+p)}{1+2aD-3a(1-3c_s^2)(\varepsilon+p)} .
$$

**Conformal matter, exact.** For $p=\varepsilon/3$, let $b=\beta\kappa\varepsilon/3$ and $s=\sqrt{1-16b}$. Then

$$
P_1=-\tfrac13P_0=\frac{1-s}{8\beta},\quad \sigma_0=\frac{1+3s}{4},\quad \sigma_1=\frac{5-s}{4},\quad \tilde\Delta=\frac{4(1-s)}{3(3+s)} .
$$

* For $\beta>0$ a real solution exists only for $\beta\kappa\varepsilon\le3/16$. This root and ceiling are known from bouncing cosmologies (Barragán & Olmo 2010: $\kappa\rho_{\max}=3R_P/16$ for radiation, with $\beta=1/R_P$). New here is the Einstein-frame effective fluid, whose pressure vanishes at $\beta\kappa\varepsilon=0.18$, before the fold.
* For $\beta<0$ one needs $|\beta|\kappa\varepsilon<9/2$.
* Expanding gives $\tilde\Delta=\tfrac89\beta\kappa\varepsilon(1+2\beta\kappa\varepsilon)$.

### 4.3 Frame invariance of momentum density

Off the diagonal $\kappa\tilde T^\mu{}_\nu=|\Omega|^{-1/2}\kappa T^\mu{}_\nu$, hence, exactly,

$$
\sqrt{-q}\,\tilde T^0{}_i=\sqrt{-g}\,T^0{}_i ,
$$

so angular momenta and the crustal moment of inertia computed in the Einstein frame are physical.

### 4.4 Contact interaction from eliminating the connection

For $\beta=0$ the action is exactly

$$
S=\int\sqrt{-q}\Big[\frac{R(q)}{2\kappa}-\frac{(\phi-1)^2}{8\alpha\kappa\phi^2}\Big]+S_m[q/\phi,\psi],\qquad \phi_*=1-2\alpha\kappa T .
$$

In general, writing $g=q+h$ and eliminating the algebraic auxiliary metric gives, to first order,

$$
h_{*\mu\nu}=2\alpha\kappa T q_{\mu\nu}+2\beta\kappa T_{\mu\nu},\qquad
S_{\rm eff}=S_{\rm EH}[q]+S_m[q]+\int\sqrt{-q}\,\frac{\kappa}{2}\big(\alpha T^2+\beta T_{\mu\nu}T^{\mu\nu}\big).
$$

For slow particles $T_i=-m_in_i$ and $T_{i\mu\nu}T_j^{\mu\nu}=m_in_im_jn_j$, so distinct particles interact through

$$
V_{ij}(\mathbf r)=-(\alpha+\beta)\,\kappa\,m_im_j\,\delta^3(\mathbf r).
$$

**Measurability criterion.** At tree level and first order, this operator is absorbed wherever the matching matter coupling is fitted to data, as for nucleons. It is measurable only where a symmetry fixes the coupling: lepton universality ($\alpha+\beta$), lepton contact operators at colliders ($\beta$), and photons (a null result, since $T=0$).

### 4.5 Lepton-universality bound

A contact $-C\delta^3(\mathbf r)$ and a finite charge radius both shift S levels through $|\psi(0)|^2$, so

$$
\delta\langle r^2\rangle_\ell=-\frac{3(\alpha+\beta)\kappa\,m_\ell M}{2\pi Z\alpha_{\rm em}} .
$$

The same holds for the form-factor slope in Born approximation. The electronic-minus-muonic difference of the same squared radius, $d_e-d_\mu=\Lambda(\alpha+\beta)$, has lever

$$
\Lambda=\frac{3\kappa M(m_\mu-m_e)}{2\pi Z\alpha_{\rm em}}=1.088\,\frac{M/m_p}{Z}\ \mathrm{fm^2\,km^{-2}},
$$

with $M\to M_A-M_B$ for isotope differences. The levers are $+1.088$ (H), $+1.087$ (D−H), $-0.533$ ($^3$He−$^4$He) and $+2.162$ ($^4$He). The estimator is $(\alpha+\beta)_i=(d_e-d_\mu)/\Lambda_i$, $\sigma_i=\sqrt{\sigma_{d_e}^2+\sigma_{d_\mu}^2}/|\Lambda_i|$, combined by inverse variance, with a $\chi^2$ consistency test.

* The shared muonic-hydrogen input correlates the H and D−H estimates by $c\simeq-0.035$; larger correlations are handled by generalised least squares (cell 75).
* The data-selection envelope is set by the CODATA 2022 electronic-only radius, 0.8529(43) fm. With it, the H-only result is $(18.7\pm6.8)$ or $(19.1\pm6.8)\times10^{-3}$ km², depending on the muonic radius: CREMA 0.84087(39) fm, or 0.84060(39) fm from Pachucki et al. 2024. The envelope is therefore $\lvert\alpha+\beta\rvert<0.033$ km². The nuclear mass distribution regulates the $\delta$ function, which changes the energy by a relative $O(2\langle r\rangle_N/a_\ell)$: about $5\times10^{-3}$ (muonic H) and $2\times10^{-2}$ (muonic $^4$He$^+$).

### 4.6 Neutron stars

**Structure.** In log-enthalpy $h$, with $dh=d\tilde p/(\tilde\varepsilon+\tilde p)$,

$$
\frac{dr}{dh}=-\frac{r(r-2m)}{m+4\pi r^3\tilde p},\qquad \frac{dm}{dh}=4\pi r^2\tilde\varepsilon\,\frac{dr}{dh},
$$

together with:

* the tidal equation for $y=rH'/H$ (with $\tilde c_s^2$), giving $k_2$ and $\Lambda_{\rm t}=2k_2/(3C^5)$;
* the Hartle frame-dragging equation $(r^4j\bar\omega')'+4r^3j'\bar\omega=0$, giving $I=J/\Omega$;
* the crustal inertia $\Delta I_c=\tfrac{8\pi}{3}\int_{\rm crust}r^4(\tilde\varepsilon+\tilde p)e^{(\lambda-\nu)/2}\bar\omega/\Omega\,dr$.

**EOS prior.** An SLy crust, then a chiral-EFT band to $1.1n_0$, then a Gaussian process for ${\rm logit}\,c_s^2$ at 14 nodes in $\ln n$ with marginalised hyper-parameters.

**Likelihood.** Every factor, with the flat population mass prior on $[1\,M_\odot, M_{\max}]$:

| factor | input | form |
|---|---|---|
| NICER J0030, J0740, J0437 | split-normal summaries (Brandes et al.) | $\int dM\,\mathcal N(M)\mathcal N(R(M))/(M_{\max}-1)$ |
| GW170817 | $\tilde\Lambda_{\rm t}=300^{+420}_{-230}$ (90%), $\mathcal M=1.186\,M_\odot$ | split normal with $\sigma_\pm=420/1.645,\ 230/1.645$, averaged over 24 $q\in[0.73,1]$ |
| J0348, J1614 (baseline) | $2.01\pm0.04$, $1.908\pm0.016\,M_\odot$ | $[\Phi((M_{\max}-M_k)/\sigma_k)-\Phi((1-M_k)/\sigma_k)]/(M_{\max}-1)$ |
| J0952 ("+J0952") | $2.35\pm0.17\,M_\odot$ | same as above |
| pQCD ("+pQCD") | $\mu_H=2.6$ GeV, $X\in[1,4]$ | fraction of $\ln X$ for which the TOV centre is connectable; applied to the physical EOS |
| Vela ("+Vela G") | G = 1.6% or 7% | $\tfrac14\sum_{m\in\{1.2,1.4,1.6,1.8\}}\Theta(\Delta I_c/I\vert_m-G)$ |

**Evidence and Bayes factor.** Over $N$ draws common to all couplings,

$$
Z(\alpha)=\frac1N\sum_k\mathcal L(d|E_k,\alpha),\qquad B(\alpha)=\frac{Z(\alpha)}{Z(0)} .
$$

If $\Phi_\alpha$ maps a physical EOS to its effective EOS and $r_\alpha=\frac{d\pi_\alpha}{d\pi_0}\frac{w_{\rm pQCD}\circ\Phi_\alpha^{-1}}{w_{\rm pQCD}}$, then

$$
B(\alpha)=\mathbb E_{\rm GR\ posterior}[r_\alpha],\qquad \inf r_\alpha\le B(\alpha)\le\sup r_\alpha .
$$

This holds when $\Phi_\alpha$ is injective, when $\Phi_{\alpha\#}(w_{\rm pQCD}\pi)\ll w_{\rm pQCD}\pi$ on the astrophysical support, and when $r_\alpha$ carries the indicator of effective EOS that give a stable star. Under these conditions the stellar evidence measures a change of prior density, not gravity. Gravity-independent constraints on the physical EOS (pQCD, crust physics) break the degeneracy only through $r_\alpha$, and the numbers above show the effect is weak. The evidences themselves are computed directly, without assuming these conditions.

**Crust condition.** Static, stable crusts need $d\tilde p/d\varepsilon=c_s^2-(\alpha+\beta)\kappa\varepsilon+\dots>0$, i.e. $\alpha+\beta<\min_{\rm crust}c_s^2/(\kappa\varepsilon)$. For $\beta=0$ the exact condition is $\alpha\kappa[(1-3c_s^2)(\varepsilon+p)-2c_s^2(\varepsilon-3p)]<c_s^2$.

**Crustal-inertia relation** (thin crust):

$$
\frac{\Delta I_c}{I}\simeq\Big(\frac{\Delta I_c}{I}\Big)_{\rm GR}\Big(1-\frac{\alpha+\beta}{\ell_t^2}\Big),\qquad \ell_t^2=\frac{2p_t}{\kappa\varepsilon_t^2}.
$$

### 4.7 Infrared Lagrangian

For $f=R-2\Lambda-\mu^4R^{-n}$, $F=f_R=1+n\mu^4R^{-n-1}$ and the trace equation $FR-2f=\kappa T$ fix $F(\rho)$. In the quasi-static, sub-horizon limit (from the metric-frame Palatini equations),

$$
\ddot\delta+2H\dot\delta=4\pi G\rho\,\mu\,\delta,\qquad \mu=\frac{1-k^2/k_J^2}{F},\qquad \Sigma=\frac1F,\qquad k_J^2=\frac{\kappa a^2}{-F_\rho}.
$$

* $F_\rho<0$, so clustering can only decrease.
* $d\ln F/d\ln a=9\Omega_m(a)(aH/k_J)^2$.
* $k_J$ is smallest at $(1+z_J)^3=8\Omega_\Lambda/[(3n+4)\Omega_m]$, which gives $z_J=0.351$ for $n=1$.
* Inside dense matter with $T>0$, $F\simeq(\kappa T)^2/(9\mu^4)$ for $n=1$, a non-analytic GR limit.

### 4.8 Compactness bound in GR (by-product)

For causal EOS with $p\le\varepsilon/3$ everywhere, and assuming the stiffest admissible EOS $p=\min(\varepsilon-\varepsilon_s,\varepsilon/3)$ is extremal, every stable star has

$$
\frac{GM}{Rc^2}\le0.3126\iff R\ge4.72\,\mathrm{km}\,(M/M_\odot).
$$

For causality alone the same procedure gives 0.3541.

## 5. Data

**Downloaded by the notebook:**

| Data | Source |
|---|---|
| DESI DR2 BAO (`desi_gaussian_bao_ALL_GCcomb_mean.txt`, `_cov.txt`) | https://github.com/CobayaSampler/bao_data |
| SDSS DR16 LRG and QSO $f\sigma_8$ blocks | https://github.com/CobayaSampler/bao_data |
| Pantheon+ distances and STAT+SYS covariance | https://github.com/PantheonPlusSH0ES/DataRelease (`Pantheon+_Data/4_DISTANCES_AND_COVAR/`) |
| Unified EOS tables SLy, BSk19, BSk20, BSk21, FPS | https://raw.githubusercontent.com/lscsoft/lalsuite-archive/master/lalsimulation/src/LALSimNeutronStarEOS_<NAME>.dat |

**Published values entered in the notebook:** charge radii (electronic H spectroscopy as tabulated in arXiv:2604.15546, muonic H, D, He, PRad, CODATA 2022), NICER mass–radius summaries, GW170817 $\tilde\Lambda_{\rm t}$, pulsar masses, the pQCD constraint, compressed Planck 2018 priors, 6dFGS $f\sigma_8$, and the DES Y6, KiDS-Legacy and ACT DR6 lensing $S_8$ values. References are in the paper. Optional: posterior-sample files for NICER and GW170817 (Section 8, check 4).

## 6. Numerical methods and validation

**Effective-fluid map.** Newton iteration with an analytic 2×2 Jacobian and continuation from GR in eight steps. On 4 000 random states it reproduces GR to $3\times10^{-13}$, the closed $f(R)$ form to $9\times10^{-13}$, and $\tilde T=T/F$ to $10^{-13}$. The first-order residual scales as coupling² (fitted exponents 1.99, 1.99, 1.97).

**Frame invariance.** A full 4×4 solve for boosted fluids satisfies the identity to $2\times10^{-15}$.

**Stars.** The batched GPU solver (enthalpy form, RK4, crust and core integrated separately) is checked against an independent scipy radius integration:

* $M_{\max}$ to 0.1%;
* $R_{1.4}$ to 0.05 km;
* $\Delta I_c/I$ to 0.02 percentage points;
* $I$ from $J/\Omega$ against the volume integral to $3\times10^{-7}$.

**Cosmology.** Checked offline against CAMB:

* distances to $2\times10^{-5}$;
* growth ratios to $10^{-3}$;
* halofit to 0.4% up to $k=30\,h$/Mpc;
* Limber CMB lensing to 0.5%.

## 7. Results reproduced by the notebook

| Quantity | Value |
|---|---|
| $\alpha+\beta$, four atomic systems | $(0.8\pm2.0)\times10^{-3}$ km², $\chi^2=2.82$ / 3 dof |
| hydrogen only | $(3.1\pm3.7)\times10^{-3}$ km² |
| largest 95% $\lvert\alpha+\beta\rvert$ over statistical variants (cell 75) | 0.0208 km² |
| 95% envelope over data selections | $\lvert\alpha+\beta\rvert<0.033$ km² (0.0324 before rounding) |
| H–(D−H) correlation from the shared muonic-H input | $c\simeq-0.035$ |
| contact model vs common constant shift | $\chi^2=2.82$ vs 0.89 (3 dof each) |
| neutron stars, astro + pQCD, 95% | $-7.55<\alpha<2.19$ km² |
| evidence change over $-4\le\alpha\le2$ km² | $\lvert\log_{10}B\rvert<0.48$ (0.41–0.56 under four reweighted EOS priors) |
| largest bootstrap 2σ error on $\log_{10}B$ (whole scan) | 0.47 |
| valid-star fraction at α = 2.25 / 2.4 km² | 1.00 / 0.00 (upper edge set by the crust) |
| crust threshold (ensemble crust / SLy / FPS) | 2.409 / 3.35 / 0.73 km² |
| crustal length $\ell_t^2$ (ensemble / SLy) | 3.41 / 3.88 km² |
| Ricci-squared ceiling, conformal matter | $\beta\kappa\varepsilon\le3/16$ |
| infrared sector vs ΛCDM (baseline / linear-only) | $\Delta\chi^2=0.79$ / 0.27; $k_J^*>1.47$ / 0.552 $h$/Mpc |
| conditional compactness bound | $C\le0.3126$ |

## 8. Pre-registered tests and referee checks

**Pre-registered (cells 61, 64):**

* P1: no stars above the crust threshold.
* P2: crustal-inertia slope within 15% of $-1/\ell_t^2$.
* P3: $\max\lvert\log_{10}B\rvert<0.5$ on $[-4,2]$.
* P4: crust/$M_{\max}$ sensitivity > 30.
* Q1–Q5: contact coefficient, frame invariance, crust robustness, laboratory bound, consistency.

All pass. One development-stage test, a sign change of $dM_{\max}/d\alpha$ with the central trace anomaly, failed and was replaced before the full run.

**Referee checks (cells 75–79).** Cells 75–77 run on a CPU from the published checkpoints and write `pj_ckpt/referee/*.json`. The results below are those obtained on molab and quoted in the paper.

**1. Atomic robustness (cell 75, < 1 s).** The levers are recomputed from the constants (H 1.0884, D−H 1.0873, ³He−⁴He −0.5330, ⁴He 2.1619 fm² km⁻²). The electronic mean, 0.84287 ± 0.00234 fm with χ² = 22.4 for 6 dof, is reproduced.

| variant | H only (10⁻³ km²) | combined (10⁻³ km²) | χ² |
|---|---|---|---|
| baseline | +3.1 ± 3.7 | +0.8 ± 2.0 | 2.82 |
| random effects (DerSimonian–Laird) | +8.1 ± 5.2 | +1.2 ± 2.2 | 4.36 |
| jackknife, lowest (drop Brandt 2022) | +1.7 ± 3.1 | — | — |
| jackknife, highest (drop Maisenbacher 2026) | +9.7 ± 5.7 | — | — |
| H/D−H correlation c = 0.5 | — | +0.3 ± 2.3 | 2.95 |
| H/D−H correlation c = 0.9 | — | −1.0 ± 2.3 | 4.25 |
| muonic errors ×2 (D−H, ³He−⁴He, ⁴He) | — | +2.1 ± 2.7 | 0.97 |
| without H / D−H / ³He−⁴He / ⁴He | — | −0.2 ± 2.4 / +0.9 ± 2.8 / +1.9 ± 2.1 / +0.4 ± 2.1 | 2.25 / 2.81 / 0.44 / 2.48 |

The largest 95% value of $\lvert\alpha+\beta\rvert$ over these variants is 0.0208 km². The contact model gives χ² = 2.82 and a common constant shift gives 0.89 (3 dof each), so the $M/Z$ pattern is a prediction, not yet a detection; the bound does not depend on it.

**2. Closed-form checks (cell 76, about 10 s).**

| check | numerical | exact |
|---|---|---|
| conformal closed form vs root-finding | 2.7 × 10⁻¹² | round-off |
| fold, βκε | 0.187500 | 3/16 |
| effective pressure zero, βκε | 0.180000 | 0.18 |
| σ₁ at βκε = −4.49 / −4.51 | +0.0013 / −0.0013 | sign change at −9/2 |
| second-order coefficient of Δ̃ | 2.006 | 2 |
| first-order residual exponents (α, β, mixed) | 2.00, 2.01, 1.99 | 2 |
| second-order residual exponent | 2.99 | 3 |
| exact sound speed vs finite difference | 5 × 10⁻¹⁰ | round-off |
| Gaussian Bayes model: Monte Carlo / analytic / posterior mean of $r_\alpha$ | 0.7402 / 0.7422 / 0.7422 (and two more couplings) | equal |
| finite size, muonic H / ⁴He⁺: exact vs $2\langle r\rangle/a$ | 4.90 vs 4.91 × 10⁻³; 2.12 vs 2.15 × 10⁻² | equal at leading order |

**3. Neutron-star evidence (cell 77, about 1 min).** It uses 20 000 common EOS draws and 200 bootstrap resamples. The largest 2σ error on $\log_{10}B$ over the whole scan is 0.47.

| EOS prior (importance reweighting) | prior ESS | 95% low α | 95% high α | max $\lvert\log_{10}B\rvert$ on [−4, 2] |
|---|---|---|---|---|
| baseline | 20 000 | −7.55 | +2.19 | 0.48 |
| softer (small $R_{1.4}$) | 1 354 | −7.78 | +2.16 | 0.41 |
| stiffer (large $R_{1.4}$) | 13 857 | −7.23 | +2.20 | 0.56 |
| heavier ($M_{\max}>2.2$) | 11 693 | −7.91 | +2.17 | 0.44 |
| lighter ($M_{\max}<2.2$) | 9 642 | −6.74 | +2.20 | 0.54 |

The fraction of EOS that still give a valid star is 1.00 at α = 1.5, 2.0 and 2.25 km², and 0.00 at α = 2.4, 2.5 and 3.0 km². The upper edge is set by the crust condition, not by the data.

**4. Sample-based likelihoods (cells 78–79, GPU).** These re-run the α scan (8 000 EOS × 9 couplings, the same draws) with KDEs of the published NICER and GW170817 posterior samples. Before running, put these files in `ns_samples/`:
- two-column $(M\,[M_\odot], R\,[\mathrm{km}])$ files `J0030.txt`, `J0740.txt` and `J0437.txt`;
- `GW170817.txt`, with a column of $\tilde\Lambda$ or columns `m1 m2 lambda1 lambda2`.

The sample files are public but not redistributed here.

## 9. Checkpoint layout on Hugging Face

Checkpoints are available at [https://huggingface.co/datasets/APassionateResearchers/palatini-molab-runs/tree/main](https://huggingface.co/datasets/APassionateResearchers/palatini-molab-runs/tree/main).

```
latest/pj_ckpt/                       newest state (use this to restore)
  RESULTS_<tag>.json, PROFILE_<tag>.json, prof_*.json, lcdm_*.json, robust_*.json, mcmc_*.npz   Part I
  part2/batch_*.npz, PART2_RESULTS.json                                                       Part II
  part3/<variant>/..., PART3_RESULTS.json                                                     Part III
  part4/PART4_RESULTS.json, PART5_FORECAST.json, PART6_COMPACTNESS.json, PART8_*.json, *.png   Parts IV-VIII
  part7/a<alpha>_b<beta>_batch<k>.npz                                                         Part VII per-point results
runs/<run-id>/pj_ckpt/                immutable snapshot of each run
paper/figures/, paper/figures.zip     paper figures under the LaTeX file names
```

Each `part7/*.npz` holds, per EOS draw:

* `ll`, `ll_bw` (with PSR J0952), `pqcd`;
* `Mmax`, `R14`, `L14`, `I14`;
* `f12`, `f14`, `f16`, `f18` ($\Delta I_c/I$ at 1.2–1.8 $M_\odot$);
* `Dc`, `valid`, `n_last`.

The file name encodes $(\alpha,\beta)$ and the batch, so files are unambiguous even if the grid changes.

## 10. Environment variables

| Variable | Effect |
|---|---|
| `SMOKE=1` | tiny sizes everywhere; checkpoints go to `pj_ckpt_smoke/` |
| `RUN_HEAVY=1`, `RUN_MCMC=0/1`, `RUN_ROBUST=1` | Part I profile likelihood, MCMC, robustness |
| `RUN_NS=1`, `RUN_P3=1`, `RUN_P4=1`, `RUN_P7=1`, `RUN_KDE=1` | heavy stages of Parts II, III, IV, VII and referee check 4 |
| `PJ_OVERRIDES`, `NS_OVERRIDES`, `P3_OVERRIDES`, `P4_OVERRIDES`, `P7_OVERRIDES` | JSON dictionaries that override configuration entries |
| `PF_CKPT` | checkpoint folder used by the figure, backup and referee cells (default `pj_ckpt`) |
| `HF_TOKEN` | Hugging Face token for the optional backup cells (never stored in the notebook) |

## 11. Security and anonymity

The released notebook contains no access tokens, e-mail addresses or personal paths. `make_release.py` removes them, rewrites the backup cells to read `HF_TOKEN` from the environment, and refuses to finish if anything secret-looking remains. `python make_release.py palatini_jeans_window.py --audit-hf` also scans the public dataset.

## Citation

Please cite the accompanying paper (bibliographic details to be added on publication).

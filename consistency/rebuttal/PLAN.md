# ICLR 2027 rebuttal — Submission 31739 (Pairwise Consistency Models)

Deadline for the revised PDF and the responses: **19 Nov 2026, 12:59 CET**. Scores: qyTj 1, XETn 1, aayn 2.
Main-text limit for the revision: **9 pages** (10 only for a camera-ready).

## 1. Diagnosis (confirmed)

* Pairwise CM (`consistency_models_CM.py`): student and teacher inputs share one noise draw, so the target is a
  point of the forward interpolant, not of the PF-ODE trajectory. The L2 optimum is a DDIM-like linear
  regression: no semigroup, one-step output shrunk to the mean.
* VarCM (`consistency_models_VarCM.py`, `l_long`): direct regression from x_T to x0 → E[x0 | ·], zero spread.
* `toy_gaussian_targets.py` reproduces XETn's numbers exactly (std 0.0125 instead of 1; semigroup gap 2.3e-3 for
  80→5→0) and shows SSR 0.006 with the current targets vs 0.99 with ODE-teacher targets.
* In our favour: ∇J and y *are* fed to VarCM during training in the code; Alg. 3 in the paper omits them.

## 2. Method fix

For all four variants, replace the shared-noise targets and `l_long` by CTM-style ODE-teacher targets:

* target of g(x_t, t, t') = g_EMA(Φ_teacher(x_t, t → t − Δt), t − Δt, t'), with Φ one Heun step of the
  already-trained FM (CM, VarCM) or DynFM (DynCM, DynVarCM) velocity, stop-gradient;
* boundary g(x, t, t) = x (parametrisation);
* tangent condition ∂_{t'} g(x, t, t')|_{t'=t} = v_teacher(x, t) (flow-matching anchor, answers XETn W2 and
  makes "follows the transport field" well posed); for VarCM the network is conditioned on ∇J and y;
* keep L_IC / L_phys only if defined explicitly in the paper; otherwise drop.

Ablation: old targets vs teacher targets; with / without tangent term.

## 3. Experiments (driven by `odyssey/pilot.py`, one job at a time)

| Step | What | Answers |
|---|---|---|
| p0_env, p0_inventory | environment; where the FM/DynFM checkpoints are (teachers) | — |
| gp_eval_old | submitted models, new metrics: SSR, rank histogram, RMSE/CRPS/spread vs the exact GP posterior, CRPS of the exact posterior as lower bound | qyTj W3, XETn W4, aayn W3 |
| gp_train_new | CM, VarCM, DynCM, DynVarCM with teacher targets (3 seeds if time) | qyTj W1, XETn W1–W2, aayn Q1–Q2 |
| gp_eval_new + NFE sweep | same metrics, NFE per entry, ensemble size | XETn W5, aayn W5 |
| gp_ablation | old vs new targets, tangent on/off; consistency on unsampled (t, t') triplets | XETn W5, aayn Q4 |
| ssh_train_new / ssh_eval | SSH_GF, training window restricted to 2013-01-02 → 2013-09-30, λx in degrees | XETn W5 |
| sic_* | SIC, only if the GPU budget allows | — |

Out of reach in 10 days, said plainly in the responses: SDA / DiffDA on their own benchmarks (qyTj W2). CTM is
discussed as the pairwise-target ancestor of our corrected CM (aayn W4); the contribution claimed is VarCM /
DynCM / DynVarCM, not the pairwise primitive.

## 4. Paper edits (structure unchanged, 9 pages)

* Abstract / intro: remove "forecasting"; "uncertainty-aware" only if the new SSR supports it.
* Sec. method: corrected targets, tangent condition; Alg. 3–5 rewritten (∇J, y, teacher step; L_IC and L_phys
  defined or removed); mean vs mode wording (XETn W2).
* Related work: CTM, Boffi et al. (progressive self-distillation), shortcut models, MeanFlow, GCTM, SDA, FlowDAS,
  DiffDA.
* Experiments: new tables (calibration columns, exact-posterior reference on GP), NFE-vs-accuracy figure moved to
  the main text, all hyper-parameters / seeds / splits / ensemble size in the appendix.
* Discussion: convergence and identifiability of VarCM, what happens when J or g do not converge (aayn W1).
* App. F: Eqs. 61–62 corrected (G = M⁻¹(I − QP⁻¹), Σ_back general form, stationary case as special case); either
  derive DynVarCM's linear-Gaussian limit or reduce the claim at L407–410.
* Figures: readable fonts (Fig. 2 first).

## 5. Responses to reviewers

One response per reviewer, plus a confidential note to the AC (XETn's "Questions" refer to another paper:
Plate, AMG, ETH, hard gates). Tone: acknowledge the target flaw directly, show the toy result, describe the fix and
the new results, list what changed in the PDF with line references.

## 6. Calendar

| Date | Code / runs | Paper / responses |
|---|---|---|
| 9–10 Nov | pilot, toy check (done); teacher-target loss for CM, VarCM; new metrics | — |
| 11–12 Nov | DynCM, DynVarCM; gp_eval_old, gp_train_new | App. F, algorithms, related work, claims |
| 13–15 Nov | gp_eval_new, ablation; SSH | NFE figure, fonts, appendix details |
| 16–17 Nov | SIC if possible; tables and figures | results into the paper; discussion |
| 18 Nov | — | responses + AC note |
| 19 Nov (morning) | — | page check, upload |

## 7. Decisions and results log

* 9 Nov — VarCM positioned as a deterministic (conditional-mean) variational solver, with the tangent condition
  (VarCM-B, `VarCMT`). Uncertainty comes from the corrected CM / DynCM. Stochastic VarCM: existing checkpoints
  evaluated (SSR 0.02–0.06, collapsed); stated as a limitation, Langevin anchor + energy-score composition as
  future work (option B, only if time remains).
* 9 Nov — GP, CMT (600 epochs, warm start from CM): RMSE 0.198 (CM 0.229, FM 0.198), CRPS on observed pixels
  0.021 (CM 0.109, FM 0.016), SSR 0.82 (CM ≈ 0.11), CRPS all pixels 0.110. OI RMSE 0.225: not the exact posterior.
* 9 Nov — DynCMT (DynFM teacher, no L2 data anchors) added for GP.
* 10 Nov — VarCMT (tangent condition, lambda 1, warm start + curriculum restart from N = 5): RMSE 0.360 vs 0.230
  for VarCM, lambda_x 130 vs 32. Not used. Cause not separated (curriculum restart vs tangent term).
* 10 Nov — User: a teacher-free, trajectory-free posterior sampler in the variational branch is needed to
  publish. VarCRPS: VarCM with fair-CRPS ensemble anchors (8 draws of x_T per (x0, y)), fixed grid, warm start.
  Same idea to be ported to VarDynCM if the GP result is good. Priority over DynCMT training.
* 10 Nov — User: finish all methods on GP before SSH; keep the original method names. VarCM and VarDynCM are
  now the corrected (CRPS) versions; the submitted ones are VarCM_v1 / VarDynCM_v1 (notebooks *_v1_spde, results
  *_v1_metrics*). Spatial correlations: the CRPS anchor includes the fair CRPS of spatial increments at lags
  1, 2, 4, 8 px (both axes) and of temporal increments (variogram-type terms), checked on synthetic fields:
  pixel CRPS alone cannot tell correct from white/too-smooth fields with the same marginals, the increment
  terms rank the correctly correlated field first. VarDynCM: CRPS of 8 rollouts on the sampler's frame grid +
  self-consistency of each jump (no L2 anchors).
* 11 Nov — GP, CRPS versions (20 members, full test set):
  VarDynCM (CRPS rollouts, 100 ep): RMSE 0.231 (v1 0.271), SSR 0.45 (v1 0.03), CRPS all px 0.145 (v1 0.165),
  rank dev 1.48 (v1 3.02); training CRPS still decreasing -> continued to 200 epochs (gp_vardyncm_more).
  VarCM (CRPS anchor after the teacher's h_t, 200 ep): SSR 0.08 (v1 0.05), RMSE 0.236 (v1 0.230), L_long flat
  from epoch ~20 -> still collapsed. Cause: the CRPS acts on student(h_t) where h_t comes from the EMA teacher
  without gradient; a collapsed teacher gives h_t without information on x_T. Fix: also score the student's
  direct jump x_T -> 0 (crps_direct); run crps2 warm-started from crps1. crps1 results kept as VarCM_crps1_*.
* 12 Nov — DynCMT (DynFM teacher): RMSE 0.229, CRPS all 0.127, SSR 0.96, rank dev 0.07 (= DynFM). VarCM crps2
  (direct CRPS term, 150 ep): RMSE 0.219, lambda_x 21, CRPS all 0.147, SSR 0.18; L_long still decreasing.
  User: option A, continue VarCM to 300 epochs (gp_varcm_more) after VarDynCM, then SSH.

### 2026-11-13 — second CRPS stage for VarCM and VarDynCM (user decision: levers 2 + 3, SSH after)
* Self-consistency weight 0.1 instead of 1 (`LAMBDA_SELF`; VarCM `lambda_pair`, VarDynCM `lambda_semi`): in the
  VarDynCM run, L_comp reached 0.33, oscillated and fought the CRPS (IC collapse and checkerboard at 200 epochs).
* Checkpoint chosen on the **validation CRPS** of the ensemble (`SELECT_BY_VAL`, the training score on 50–100
  validation batches with a fixed seed per batch), not on the last epoch nor on VarDynCM's old `val_loss`
  (deterministic IC MSE, which rewards collapse).
* VarDynCM_ls01: 100 epochs from the submitted VarDynCM. VarCM_ls01: 100 epochs from VarCM (crps2, 300 epochs).
  Same notebooks (`RUN_TAG=_ls01`, own LOG_DIR and figure folder); methods.yaml entries `*_ls01`.
* Plan: gp_ls01_smoke → gp_vardyncm_ls01_train (~7 h) → gp_varcm_ls01_train (~4 h) → SSH.

### 2026-11-13 evening — ls01 runs: no gain; diagnosis and VarCM v3
* VarCM_ls01: validation CRPS flat for 100 epochs (best epoch 43); test identical to VarCM (SSR 0.20).
  VarDynCM_ls01: validation CRPS still decreasing at epoch 99; test worse (SSR 0.08). Not used in the paper.
* Two suspects in VarCM: (i) CRPS on 1- and 2-jump maps, but evaluation with a 15-step deterministic chain;
  (ii) L_short is still an L2 (pseudo-Huber) regression on x0 from the forward interpolant (mean collapse).
* gp_nfe_diag: spread/SSR versus the number of sampler steps (VarCM, VarDynCM).
* VarCM_v3: SHORT_CRPS (no L2 target left) + ROLLOUT_STEPS (CRPS on the sampler's own K-step chain,
  K in {3,5,8,15}, gradient through the last jump), 100 epochs from crps2, validation-CRPS selection.
  gp_varcm_v3_smoke runs automatically; gp_varcm_v3_train waits for approval after the diagnostic.

### 2026-11-14 — results of v3 and of the frame-grid evaluation
* VarCM_v3 (CRPS short anchor + CRPS on the sampler chain): no change (SSR 0.19, CRPS 0.149); validation CRPS
  flat at ~1.25 as in crps2 and ls01. Three objectives, same plateau: the limit is not in the loss terms.
* VarDynCM sampled on its training frame grid (1 -> IC, one jump per frame): VarDynCM_fg RMSE 0.173, CRPS 0.100,
  SSR 0.36; VarDynCM_ls01_fg RMSE 0.180, CRPS 0.100, SSR 0.53, rank dev 0.91 (vs SSR 0.08 on 25 uniform steps).
  Spread grows along the window (small at the IC). Not consistent across grids (NFE sweep): to be stated.
* gp_vardyncm_ls01b_train (confirm): 100 more epochs of ls01 on the frame grid.

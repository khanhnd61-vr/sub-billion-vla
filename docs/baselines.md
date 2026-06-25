# Notes

## Baseline comparison

Both VLA-Adapter and SmolVLA share one structure - a **frozen pretrained VLM + a from-scratch action module
trained on LIBERO** - but differ in the VLM, the action paradigm, and (here) the data scope.

### Training objective

| | **VLA-Adapter** | **SmolVLA** |
|---|---|---|
| Objective | **L1 regression** (deterministic) | **flow matching** (generative) |
| Loss | `L1(pred_actions, gt_actions)` | `MSE(u_t, v_t)` on the velocity field |
| Net predicts | the action chunk directly | the velocity `u_t = noise - actions` |
| Noise / time input | none | yes (sampled `ε`, `t`) |
| Inference | single forward | sample noise over ~10 Euler ODE steps |
| Output nature | single mode (regression mean) | can model **multimodal** action distributions |

> The SmolVLA training loss is a velocity-MSE, **not** an action error - not comparable in scale to
> VLA-Adapter's L1. Judge both by **eval success rate**, not by comparing loss curves.

### Model architecture

| | **VLA-Adapter** | **SmolVLA** |
|---|---|---|
| VLM backbone | Prismatic MiniVLM: **DINOv2 ViT-L + SigLIP SO400M** + **Qwen2.5-0.5B** (h=896, 24L), @224px | **SmolVLM2-500M-Video-Instruct**: SigLIP + SmolLM2, **16 VLM layers used** |
| VLM treatment | Frozen **+ LoRA r64** (all-linear) | **Fully frozen**, no LoRA (`train_expert_only`) |
| Action module | **Bridge**: 24-block MLP-ResNet, per-layer *gated* cross-attn to VLM states; 64 learnable **ActionQuery** tokens; proprio projector | **Flow-matching expert**: transformer ~**0.75× VLM width**, cross-attn to VLM every 2 layers, time+noise conditioned |
| Trainable params | **~207M** (LoRA + ActionQuery + Bridge + proprio) | **~100M** (action expert only) |
| Total params | **~1.5B** (dual vision encoders dominate) | **~450M** |

### Training dataset

| | **VLA-Adapter** | **SmolVLA (this recipe)** |
|---|---|---|
| Suite scope | **LIBERO-Spatial only** (10 tasks) | **All 4 suites** (Spatial+Object+Goal+Long) |
| Source / format | openvla `libero_spatial_no_noops`, **RLDS/TFDS** | `HuggingFaceVLA/libero`, **LeRobotDataset v3.0** |
| Size | Spatial only (~10×50 demos) | 273,465 frames |
| Cameras | 2 (3rd-person + wrist) -> 512 vision tokens | 2 (`image` + `image2`) |
| Action | 7-dim (xyz + axis-angle + gripper), `bounds_q99`->[-1,1] | 7-dim **relative delta** EEF, **mean/std** |
| State | 8-dim | 8-dim, mean/std |
| Action chunk | 8 steps | 50 steps |

### Training configuration

| | **VLA-Adapter** | **SmolVLA (this run)** |
|---|---|---|
| LR schedule | **2e-4 constant** | **1e-4 cosine** + 1000 warmup -> 2.5e-6 |
| Steps | 30k | T.B.D |
| Precision | bf16 | mixed precision (AMP) |
| Adaptation | LoRA r64/α128 | full action-expert finetune (no PEFT) |
| Peak VRAM | ~17 GB @ batch 4 (grad-accum 4) | ~13 GB @ batch 64 (no grad-accum) |

> **Not apples-to-apples:** VLA-Adapter trains on *Spatial only*, while this SmolVLA recipe trains on
> *all four suites* (~16× more/broader data). Compare on **per-suite eval SR**, not headline averages.

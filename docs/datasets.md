# VLA Pre-training Datasets on Hugging Face

Datasets for sub-billion VLA pre-training. Two categories: robot-trajectory datasets (teleoperated robot demos, directly trainable) and human-video datasets (first-person human video requiring a transfer pipeline).

## Robot-trajectory datasets

Sorted ascending by dataset size.

| Dataset | What it is | Dataset size | Trained models | Format |
|---|---|---|---|---|
| [lerobot/droid_100](https://huggingface.co/datasets/lerobot/droid_100) | 100-episode sample of DROID (Franka 7-DoF), for smoke-testing the pipeline | 464 MB (exact) | smolvla_test, pi05 (community) | LeRobot v2.0 (parquet + AV1 MP4) |
| [community_dataset_v2](https://huggingface.co/datasets/HuggingFaceVLA/community_dataset_v2) | Curated crowdsourced LeRobot tabletop data, mostly SO-100/SO-101 single-arm | ~120 GB (v1 = 119.3 GB; v2 same order) | SmolVLA | LeRobot v2.1 |
| [DROID (full)](https://huggingface.co/datasets/lerobot/droid_1.0.1) | In-the-wild Franka 7-DoF, strong language annotations + camera calibration | ~2 TB (raw, per LeRobot docs) | OpenVLA, π0, RoboGene, LabVLA | LeRobot (official port) |
| [Galaxea Open-World](https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset) | Open-world homes/offices on one R1-Lite bimanual mobile manipulator, bilingual subtask labels | ~3–4 TB (227 task tar.gz; estimate) | G0, G0Plus, G0Tiny (250M, SmolVLM2) | LeRobot v2.1 (gated) |
| [RoboMIND](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) | Multi-embodiment incl. humanoid + dexterous hands (Franka, Tien Kung, AgileX, UR-5e) | ~5–7 TB (estimate; HF hosts v1.0–1.2) | RDT-1B, CrossFormer | Native HDF5 → convert |
| [AgiBotWorld-Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) | First-party data on one AgiBot G1 bimanual mobile manipulator | ~8.5 TB (exact, per dataset card) | GO-1; GigaBrain-0, X-VLA, TrajBooster | Native HDF5 → convert |
| [OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) | Aggregated meta-dataset, 22 embodiments, 60 sub-datasets | ~9+ TB (varies by version; streamable) | RT-X, Octo, OpenVLA, π0 | RLDS/TFDS (HF mirror) |
| [RH20T](https://huggingface.co/datasets/hainh22/rh20t) | Multi-robot manipulation, single-arm | ~10 TB (raw, official estimate) | - (benchmark-oriented) | LeRobot (community port) |

## Notes on sizes

- **Exact figures:** droid_100 (464 MB, read off the HF card) and AgiBotWorld-Alpha (~8.5 TB, stated on the dataset card). DROID-full (~2 TB) is the figure cited in LeRobot's own porting guide.
- **Estimates:** Galaxea, RoboMIND, OXE, and RH20T are published or order-of-magnitude estimates. These repos shard across many archives or external hosts and do not surface one clean total. RH20T and OXE swing significantly by version, so treat their position near the bottom as approximate. The takeaway is that all four sit in the same multi-TB heavyweight tier.
- **Format caveat:** for community LeRobot ports (RH20T, OXE sub-datasets), verify the `codebase_version` in `meta/info.json` matches your installed `lerobot` version. A mismatch there is a common, confusing load-time failure.

BridgeData V2 is omitted: it has no canonical first-party HF repo and ships via its project site or bundled (outdated) inside OXE.

## Human-video datasets

These pre-train a VLA on human first-person video (often with 3D hand / MANO reconstruction), then transfer to robots. Filtered to datasets with a confirmed HF dataset link; others are noted below. None are LeRobot-format or robot-action-labeled, so all require a pose-reconstruction / retargeting pipeline before use with SmolVLA.

| Dataset | What it is | Size | Access / License | Used by (VLA pretraining) |
|---|---|---|---|---|
| [VITRA-1M](https://huggingface.co/datasets/VITRA-VLA/VITRA-1M) | Human-hand VLA dataset: 1.2M short episodes with segmented language, corrected camera params, and MANO 3D hand reconstructions (each episode a single .npy metadata file) | 1.2M episodes (metadata only; source video from Ego4D/Ego-Exo4D) | HF direct; inherits source-video licenses | VITRA-VLA-3B |
| [Egocentric-10K](https://huggingface.co/datasets/builddotai/Egocentric-10K) | Build AI factory-worker video, first collected exclusively in real factories | 10,000 hr, 2,153 workers, 1.08B frames, 1080p 30fps, 16.4 TB | HF direct, **Apache 2.0 (commercial OK)**, WebDataset, streamable, ungated | Industrial manipulation / IL baselines |
| [Egocentric-100K](https://huggingface.co/datasets/builddotai/Egocentric-100K) | 10x scale-up of Egocentric-10K | 100,000+ hr, 14,228 workers, 10.8B frames | HF, **Apache 2.0**, WebDataset, stream-only; **gated (must accept conditions)** | Foundation-model pretraining |

Notes: Among these, only VITRA-1M and Egocentric-10K are ungated direct downloads; Egocentric-100K requires accepting access conditions. The Build AI sets are the rare clean-commercial-license option in this category, but carry no robot action labels (the "action" is reconstructed hand pose / camera motion). VITRA-1M is the most VLA-ready since it ships MANO annotations and has a published 3B recipe, but it only contains metadata - you supply the source video from the gated Ego datasets.

**Excluded (no confirmed HF dataset link):**
- **EgoDex** - 829 hr, 338K episodes, 2.0 TB; distributed via Apple CDN + GitHub (`apple/ml-egodex`), CC-BY-NC-ND (non-commercial). Used by Being-H0, H-RDT.
- **Ego4D** - 3,700+ hr; consortium CLI (`facebookresearch/Ego4d`), license approval (~48 hr). Substrate for VITRA and many others.
- **Ego-Exo4D** - 1,286 hr paired ego+exo; consortium CLI, license approval (~2 days). Substrate for VITRA, EgoScale.
- **EgoVerse** - 1,362 hr, 80K episodes, 1,965 tasks; project release (arXiv 2604.07607), iPhone collection supported.
- **Egocentric-1M** - Build AI's ~1M-hour set, announced on HF under Apache 2.0, but no `builddotai/Egocentric-1M` repo could be confirmed at time of writing (verified Build AI HF repos are only 10K and 100K).

## Bottom line: first two datasets to try

1. **`lerobot/droid_100`** - validate the full pre-training loop. At 464 MB it downloads in seconds, it is official LeRobot format (zero conversion), single clean Franka embodiment with language labels, and HF already lists a community SmolVLA model trained on it, so the path is known to work end-to-end. This is a smoke test, not a real run.

2. **`community_dataset_v2`** - the first real pre-training run. It is SmolVLA's native action space and roughly its actual pretraining corpus, so it is the most faithful reproduction starting point and still requires zero conversion. Low risk, directly comparable to the published SmolVLA result.

After these two validate cleanly, the natural scale-up is **full DROID (~2 TB)**: same format and embodiment as the droid_100 smoke test, so no new conversion work, just more data. Treat the multi-TB heavyweights (Galaxea, RoboMIND, AgiBot, OXE) as deliberate later choices driven by your downstream embodiment target and licensing constraints.

Worth studying regardless of which dataset you train on: **Galaxea's G0Tiny** is a 250M SmolVLM2-backbone VLA built for Orin edge deployment - the closest published precedent to your architecture and hardware target.
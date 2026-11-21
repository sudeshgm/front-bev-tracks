# Model comparison (permissive only)

Scripts for GPU Bot to run on free GPU and push overlays back.

## Models
- TwinLiteNet (MIT) – lanes + drivable area
- Ultra-Fast-Lane-Detection (MIT) – pure lanes
- (Qwen-Drive Apache-2.0 / SparseDrive MIT later if feasible)

## Output
`testing_videos/model_compare/<model>/`

## Notes
GPU Bot pulls this branch, clones the model repos, downloads weights, runs the scripts, and commits/pushes the resulting overlays (or uploads to Drive).

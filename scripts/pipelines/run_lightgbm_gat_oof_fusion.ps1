param(
    [string]$Python = "D:\Miniconda3\envs\aml-evidence\python.exe",
    [string]$FeatureRoot = "artifacts/pit_features_fe_v2",
    [string]$GraphOof = "artifacts/graph_oof_gat_v1_local_replay/graphsage_oof_scores.parquet",
    [string]$GraphValidation = "artifacts/gat_v1_local_replay/scores/graphsage_validation_scores.parquet",
    [string]$GraphTest = "artifacts/gat_v1_local_replay/scores/graphsage_test_scores.parquet"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

& $Python -m aml_evidence_graph.training.lightgbm_baseline `
    --features $FeatureRoot `
    --output artifacts/table_lightgbm_fe_v2 `
    --model-config configs/models.gat.yaml

& $Python -m aml_evidence_graph.training.oof `
    --features $FeatureRoot `
    --output artifacts/table_oof_lightgbm_fe_v2 `
    --model table `
    --table-model lightgbm `
    --model-config configs/models.gat.yaml

& $Python -m aml_evidence_graph.training.fusion `
    --oof artifacts/table_oof_lightgbm_fe_v2/table_oof_scores.parquet `
    --oof $GraphOof `
    --validation artifacts/table_lightgbm_fe_v2/scores/table_validation_scores.parquet `
    --validation $GraphValidation `
    --components lightgbm,graphsage `
    --model-config configs/models.gat.yaml `
    --output artifacts/fusion_lightgbm_fe_v2_gat_v1

& $Python -m aml_evidence_graph.training.evaluate_fusion `
    --fusion-dir artifacts/fusion_lightgbm_fe_v2_gat_v1 `
    --test artifacts/table_lightgbm_fe_v2/scores/table_test_scores.parquet `
    --test $GraphTest `
    --features $FeatureRoot `
    --output artifacts/fusion_test_lightgbm_fe_v2_gat_v1 `
    --bootstrap-iterations 200

# Keep disk usage lean on an 8GB/12GB-free M1.
# Xet is HF's dedup chunk store; it keeps a SECOND copy of every downloaded blob.
# For our two small models that roughly doubles weight cost for zero benefit.
export HF_HUB_DISABLE_XET=1
# Do not keep .no_exist / stale revision metadata growing unbounded.
export HF_HUB_DISABLE_TELEMETRY=1

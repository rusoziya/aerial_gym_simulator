#!/usr/bin/env bash
set -euo pipefail

# Rewrites git history to permanently remove large experiment artifacts.
#
# Keeps:
# - docs/
# - resources/
# - supplementary_material/
# - aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth
# - aerial_gym/examples/dce_rl_navigation/TRAINED/HIGH_CONFIG_16ENV_2/checkpoint_p0/HIGH_CONFIG_16ENV_2_best_000025464_13041664_reward_1463.917.pth
#
# Removes from ALL commits:
# - .pth/.pt/.ckpt model artifacts (except the two whitelisted files)
# - .gif/.mp4 media outside docs/resources/supplementary_material
# - common output folders (runs/, train_dir/, train_DIR/)
#
# NOTE: This changes commit SHAs. You must force-push and coordinate with collaborators.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: not inside a git repo" >&2
  exit 1
fi

if ! command -v git-filter-repo >/dev/null 2>&1 && ! git filter-repo --help >/dev/null 2>&1; then
  cat <<'EOF'
ERROR: git-filter-repo is not installed.

Install (recommended):
  python3 -m pip install --user git-filter-repo

Or (Ubuntu):
  sudo apt-get update && sudo apt-get install -y git-filter-repo

Then re-run this script.
EOF
  exit 1
fi

KEEP_VAE="aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
KEEP_DCE="aerial_gym/examples/dce_rl_navigation/TRAINED/HIGH_CONFIG_16ENV_2/checkpoint_p0/HIGH_CONFIG_16ENV_2_best_000025464_13041664_reward_1463.917.pth"

# Safety: ensure keep files exist in current tip.
for p in "$KEEP_VAE" "$KEEP_DCE"; do
  if ! git cat-file -e "HEAD:$p" 2>/dev/null; then
    echo "ERROR: expected keep file missing at HEAD: $p" >&2
    exit 1
  fi
done

# Build regexes that exclude the keep files.
# git-filter-repo uses Python regex syntax.
RE_PY_MODEL='^(?!(docs/|resources/|supplementary_material/))(?!(aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49\.pth)$)(?!(aerial_gym/examples/dce_rl_navigation/TRAINED/HIGH_CONFIG_16ENV_2/checkpoint_p0/HIGH_CONFIG_16ENV_2_best_000025464_13041664_reward_1463\.917\.pth)$).*(\.(pth|pt|ckpt))$'
RE_PY_MEDIA='^(?!(docs/|resources/|supplementary_material/)).*(\.(gif|mp4))$'
RE_PY_OUTPUT_DIRS='^(?!(docs/|resources/|supplementary_material/)).*(?:^|/)(?:runs|train_dir|train_DIR)/'

echo "About to rewrite history to drop large artifacts."
echo "This will change commit SHAs."
read -r -p "Type 'rewrite' to continue: " confirm
if [ "$confirm" != "rewrite" ]; then
  echo "Aborted."
  exit 0
fi

# Recommended pre-step: close over any accidental staged changes.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has changes. Commit/stash before rewriting history." >&2
  exit 1
fi

# Run filter-repo in one pass.
# We remove paths matching the regexes.
# (Multiple --path-regex with --invert-paths removes the union of the matches.)

git filter-repo \
  --force \
  --path-regex "$RE_PY_MODEL" --invert-paths \
  --path-regex "$RE_PY_MEDIA" --invert-paths \
  --path-regex "$RE_PY_OUTPUT_DIRS" --invert-paths

cat <<'EOF'

Done.

Next steps (for the maintainer):
  1) Push rewritten history:
       git push --force --tags origin main

  2) Tell existing users to re-clone (or hard reset).

  3) Consider publishing removed checkpoints/GIFs as GitHub Releases (or another artifact store)
     so users can download them on demand.
EOF

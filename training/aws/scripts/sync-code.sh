#!/usr/bin/env bash
# Ship one commit of a local git repository to the host.
#   training/aws/scripts/sync-code.sh <worktree> <name> [--rev REV] [--upload-only]
# 1. `git archive REV` (default HEAD) to s3://$HOBSON_BUCKET/code/<name>.tgz: exactly the
#    files that the commit tracks (data/synthetic/ included). Untracked and ignored files
#    and uncommitted changes are not shipped. The archive also holds .hobson-manifest (the
#    file list) and .hobson-rev (the full commit SHA, which training/run_recipe.sh and
#    evaluation/jevbench/jevbench.sh record). Nothing is written into the worktree.
# 2. on the host, extracts it over /opt/hobson/code/<name> (data/ and checkpoints/ are
#    symlinks to /opt/hobson/scratch/work/<name>/{data,checkpoints} and are kept).
#    Files deleted since the last sync are removed on the host too (manifest diff).
# Host-side step alone (e.g. from another script):  /usr/local/bin/hobson-pull <name>
set -euo pipefail
source "$(dirname "$0")/common.sh"
[[ $# -ge 2 ]] || die "usage: sync-code.sh <worktree> <name> [--rev REV] [--upload-only]"
WT="$(git -C "$1" rev-parse --show-toplevel)" || die "$1 is not in a git worktree"
NAME="$2"; shift 2
UPLOAD_ONLY=0; REV=HEAD
while [[ $# -gt 0 ]]; do
  case "$1" in
    --upload-only) UPLOAD_ONLY=1; shift;;
    --rev) [[ $# -ge 2 ]] || die "--rev needs a commit"; REV="$2"; shift 2;;
    *) die "bad flag $1";;
  esac
done
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad name $NAME"
SHA=$(git -C "$WT" rev-parse --verify --quiet "$REV^{commit}") || die "$REV is not a commit in $WT"
[[ -z "$(git --no-optional-locks -C "$WT" status --porcelain --untracked-files=no)" ]] \
  || log "WARN: $WT has uncommitted changes to tracked files; they are not shipped (commit them first)"

META=$(mktemp -d "${TMPDIR:-/tmp}/hobson-code.XXXXXX"); trap 'rm -rf "$META"' EXIT
{ git -C "$WT" ls-tree -r -z --full-tree --name-only "$SHA" | tr '\0' '\n'
  printf '%s\n' .hobson-manifest .hobson-rev; } > "$META/.hobson-manifest"
echo "$SHA" > "$META/.hobson-rev"
git -C "$WT" archive --format=tar.gz --add-file="$META/.hobson-manifest" --add-file="$META/.hobson-rev" \
  -o "$META/code.tgz" "$SHA"
aws s3 cp "$META/code.tgz" "s3://$HOBSON_BUCKET/code/$NAME.tgz" --only-show-errors --region "$HOBSON_HOME_REGION"
log "uploaded $(wc -l < "$META/.hobson-manifest" | tr -d ' ') files ($(du -h "$META/code.tgz" | cut -f1)) of $SHA from $WT -> s3://$HOBSON_BUCKET/code/$NAME.tgz"
[[ $UPLOAD_ONLY -eq 1 ]] && exit 0

"$(dirname "$0")/ssm-run.sh" -q "hobson-pull $NAME"

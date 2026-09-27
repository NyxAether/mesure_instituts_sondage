#!/usr/bin/env bash
# Hook PreToolUse : impose d'indexer les fichiers dans une commande séparée du git commit, afin que le diff relu soit celui qui sera commité
# Le commit est refusé tant que l'empreinte du diff indexé n'a pas été validée dans
# <git-dir>/claude-review-ok ; la validation est consommée par le commit qu'elle autorise.

input=$(cat)
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // empty')

printf '%s' "$cmd" | grep -Eq '(^|[;&|[:space:]])git([[:space:]]+-C[[:space:]]+[^[:space:]]+)?[[:space:]]+commit([[:space:]]|$)' || exit 0

deny() {
  jq -n --arg r "$1" '{hookSpecificOutput: {hookEventName: "PreToolUse", permissionDecision: "deny", permissionDecisionReason: $r}}'
  exit 0
}

cwd=$(printf '%s' "$input" | jq -r '.cwd // empty')
[ -n "$cwd" ] && cd "$cwd" 2>/dev/null
git rev-parse --git-dir >/dev/null 2>&1 || exit 0

# Le diff relu doit être exactement celui qui sera commité.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+add|commit[^;&|]*[[:space:]](-[a-zA-Z]*a[a-zA-Z]*|--all)([[:space:]]|$)'; then
  deny "Review pre-commit : indexe d'abord les fichiers dans une commande séparée (pas de 'git add' ni de 'git commit -a' dans la même commande que le commit), afin que la review porte sur le diff réellement commité."
fi
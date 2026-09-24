#!/usr/bin/env bash
# Hook PreToolUse : impose une review du diff indexé avant tout `git commit` lancé par Claude.
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
git_dir=$(git rev-parse --absolute-git-dir 2>/dev/null) || exit 0

# Le diff relu doit être exactement celui qui sera commité.
if printf '%s' "$cmd" | grep -Eq 'git[[:space:]]+add|commit[^;&|]*[[:space:]](-[a-zA-Z]*a[a-zA-Z]*|--all)([[:space:]]|$)'; then
  deny "Review pre-commit : indexe d'abord les fichiers dans une commande séparée (pas de 'git add' ni de 'git commit -a' dans la même commande que le commit), afin que la review porte sur le diff réellement commité."
fi

git diff --cached --quiet && exit 0
hash=$(git diff --cached | sha1sum | cut -d' ' -f1)
stamp="$git_dir/claude-review-ok"

if [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$hash" ]; then
  rm -f "$stamp"
  exit 0
fi

deny "Review pre-commit requise. Relis le diff indexé (git diff --cached) : bugs, régressions, incohérences avec le code existant, fichiers ou secrets qui ne devraient pas être commités. Présente tes constats à l'utilisateur et corrige ce qui doit l'être (puis ré-indexe : la review devra alors être refaite). Une fois le diff validé, exécute : echo $hash > \"$stamp\" puis relance le même git commit."

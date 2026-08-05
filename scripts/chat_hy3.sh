#!/usr/bin/env bash
# Lightweight interactive client for the Hy3 dev server on gpu-01.
set -euo pipefail

namespace="${HY3_NAMESPACE:-llm}"
pod="${HY3_POD:-onecat-vllm-hy3-sm70-dev}"
model="${HY3_MODEL:-cyankiwi/Hy3-AWQ-INT4}"
max_tokens="${HY3_MAX_TOKENS:-512}"
temperature="${HY3_TEMPERATURE:-0.7}"

if [[ -z "${KUBECONFIG:-}" && -f "$HOME/.kube/homelab-config" ]]; then
  export KUBECONFIG="$HOME/.kube/homelab-config"
fi

for command in kubectl jq; do
  command -v "$command" >/dev/null || {
    echo "Missing required command: $command" >&2
    exit 1
  }
done

request() {
  local payload="$1"
  printf '%s' "$payload" |
    kubectl -n "$namespace" exec -i "$pod" -- \
      curl -fsSN http://127.0.0.1:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        --data-binary @-
}

messages='[{"role":"system","content":"You are a concise, helpful assistant."}]'

chat() {
  local prompt="$1" payload answer line event delta
  messages="$(jq --arg content "$prompt" \
    '. + [{role: "user", content: $content}]' <<<"$messages")"
  payload="$(jq -n \
    --arg model "$model" \
    --argjson messages "$messages" \
    --argjson max_tokens "$max_tokens" \
    --argjson temperature "$temperature" \
    '{model: $model, messages: $messages, max_tokens: $max_tokens,
      temperature: $temperature, stream: true}')"

  answer=""
  printf '\nHy3> '
  while IFS= read -r line; do
    [[ "$line" == 'data: [DONE]' ]] && break
    [[ "$line" == 'data: '* ]] || continue
    event="${line#data: }"
    delta="$(jq -r '.choices[0].delta.content // empty' <<<"$event")"
    printf '%s' "$delta"
    answer+="$delta"
  done < <(request "$payload")
  printf '\n\n'

  if [[ -z "$answer" ]]; then
    echo "No text received from the streaming API." >&2
    return 1
  fi

  messages="$(jq --arg content "$answer" \
    '. + [{role: "assistant", content: $content}]' <<<"$messages")"
}

if (($#)); then
  chat "$*"
  exit 0
fi

echo "Hy3 chat — pod $pod in namespace $namespace. Type /quit to exit."
while true; do
  read -r -p 'You> ' prompt || break
  [[ "$prompt" == '/quit' || "$prompt" == '/exit' ]] && break
  [[ -z "$prompt" ]] && continue
  chat "$prompt"
done

IP=${IP:-127.0.0.1}
PORT=${PORT:-9000}
curl -s -X POST "http://${IP}:${PORT}/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": [
      "The capital of France is",
      "Who are you?",
      "Hello, my name is Tom, I am",
      "AI future is"
    ],
    "max_tokens": 30,
    "temperature": 0
  }' |  jq # -r '.choices[0].message.content'
 
exit 0

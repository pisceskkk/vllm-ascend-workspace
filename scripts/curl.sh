IP=${IP:-127.0.0.1}
PORT=${PORT:-8091}
curl -s -X POST "http://${IP}:${PORT}/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{
    "messages": [
      {
        "content": "你是谁？",
        "role": "user"
      }
    ],
    "max_tokens": 100,
    "stop": null,
    "temperature": 0.0,
    "top_p": 0.95
  }' |  jq -r '.choices[0].message.content'

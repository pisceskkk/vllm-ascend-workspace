IP=${IP:-127.0.0.1}
PORT=${PORT:-8091}

until curl -fsS "http://${IP}:${PORT}/v1/models" >/dev/null; do
  echo "Waiting for vLLM service at ${IP}:${PORT} ..."
  sleep 2
done

echo "vLLM service is ready at ${IP}:${PORT}"

ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --summarizer example --dump-eval-details --merge-ds --debug

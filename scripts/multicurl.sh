IP=${IP:-127.0.0.1}
PORT=${PORT:-8092}

until curl -fsS "http://${IP}:${PORT}/v1/models" >/dev/null; do
  echo "Waiting for vLLM service at ${IP}:${PORT} ..."
  sleep 2
done

echo "vLLM service is ready at ${IP}:${PORT}"

TOTAL=${TOTAL:-14}

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for i in $(seq 1 "$TOTAL"); do
  (
    bash curl.sh "$i" > "$tmpdir/$i.out" 2> "$tmpdir/$i.err"
  ) &
done

wait

for i in $(seq 1 "$TOTAL"); do
  echo "===== task $i output ====="
  cat "$tmpdir/$i.out"

  if [ -s "$tmpdir/$i.err" ]; then
    echo "===== task $i stderr =====" >&2
    cat "$tmpdir/$i.err" >&2
  fi
done

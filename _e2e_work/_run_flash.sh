#!/bin/sh
# 目的：在**断网**条件下验证 MinerU 4.0 的 flash 档能否解析 pptx，
#       并搞清 "本地 server" 这套架构怎么用。
# 由外部以  docker run --network none ... sh /work/_run_flash.sh  调用。

echo "=== [1] mineru server --help ==="
mineru server --help 2>&1 | tail -30

echo
echo "=== [2] mineru server start ==="
mineru server start 2>&1 | tail -25

echo
echo "=== [3] 等 server 起来 ==="
sleep 6

echo
echo "=== [4] mineru server status ==="
mineru server status 2>&1 | tail -25

echo
echo "=== [5] 探测本地端口 ==="
(command -v ss >/dev/null && ss -ltnp) || (command -v netstat >/dev/null && netstat -ltnp) || echo "(no ss/netstat)"

echo
echo "=== [6] parse (tier=flash, 断网) ==="
timeout 900 mineru parse /work/sample.pptx --tier flash -o /work/out --json 2>&1 | tail -60
echo "parse_rc=$?"

echo
echo "=== [7] 产物树 ==="
find /work/out -type f 2>/dev/null | head -40
echo "--- du ---"
du -sh /work/out 2>/dev/null

echo
echo "=== [8] MINERU_HOME 内容 ==="
ls -la "${MINERU_HOME:-/mineru-home}" 2>&1 | head -20

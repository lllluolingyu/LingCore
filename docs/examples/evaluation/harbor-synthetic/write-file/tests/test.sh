#!/bin/bash
mkdir -p /logs/verifier
if test "$(cat /app/answer.txt 2>/dev/null)" = harbor-ok; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

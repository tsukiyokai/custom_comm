#!/bin/bash
# 同步本地代码到蓝区
rsync -avz --delete \
  --exclude='.git/' \
  --exclude='build/' \
  --exclude='__pycache__/' \
  --exclude='*.egg-info/' \
  --exclude='.sdk' \
  --exclude='.dage/' \
  --exclude='.cache/' \
  --exclude='.pytest_cache/' \
  --exclude='.claude/' \
  /Users/shanshan/repo/cann/custom_comm/ \
  bluezone:~/code/custom_comm/

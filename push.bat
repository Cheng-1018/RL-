@echo off
set http_proxy=http://127.0.0.1:7897
set https_proxy=http://127.0.0.1:7897
git add .
git commit -m "修复公式错误以及显示问题，增加仓库导航"
git push
pause
# 이 스크립트는 PowerShell에서 실행됩니다.

# Git 저장소 경로 설정
$repoPath = "c:\\Users\\ohbok\\Taipy\\sparc-malfunction-classification-develop_bak"

# 커밋 메시지 설정
$commitMessage = "Auto commit at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# Git 명령어 실행
cd $repoPath
git add .
git commit -m $commitMessage

# 푸시 (선택사항)
git push
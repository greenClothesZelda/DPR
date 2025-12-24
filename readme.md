사전준비
<br>
1. data/douwnload.py에서 nq-train, nq-dev 다운로드
<br>
2. reformatter(gpt로 만듦)로 jsonl로 변환 및 저장
```bash
#run
python main.py --config-name config hydra/job_logging=default
#background 실행
nohup python -u main.py --config-name config hydra/job_logging=default &
```
export OPENAI_BASE_URL="https://api.openai.com"
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_MODEL="gpt-4.1-mini"


python3 batch_solve_zebras.py --input /home/asif/data3/HF_cache/ZebraLogic/Zebra_Puzzle_small_320.json --output /home/asif/data3/HF_cache/ZebraLogic/pid_to_puzzle_answers_320.json --base-url http://localhost:8000 --model your-model-name --limit 5

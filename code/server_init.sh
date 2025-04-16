apt-get update
apt-get upgrade -y
apt-get install python3.12 -y
apt-get install python3-pip -y
apt-get install python3-venv -y
apt-get install git -y

cd /srv
git clone https://github.com/AlJ95/master-thesis-alj95.git
cd master-thesis-alj95/code/ragnroll_project

# Let the user input the API keys
echo "Enter your OpenAI API key:"
read OPENAI_API_KEY
echo "Enter your OpenRouter API key:"
read OPENROUTER_API_KEY

cp .env.example .env
sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_API_KEY/" .env
sed -i "s/OPENROUTER_API_KEY=.*/OPENROUTER_API_KEY=$OPENROUTER_API_KEY/" .env

docker compose --env-file=.env up -d

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# environment variables
# derived from python -m ragnroll run-evaluations ./configs/baselines/llm_config.yaml ./data/processed/config_val/evaluation_data.json ./data/processed/config_val/corpus ./output.csv --no-baselines --test-size=30
export config_path=./configs/baselines/llm_config.yaml
export evaluation_data_path=./data/processed/config_val/evaluation_data.json
export corpus_path=./data/processed/config_val/corpus
export output_path=./output.csv
export test_size=30
export experiment_name=configuration_validation

# wait 30 seconds for the server to be ready
sleep 30

screen -dmS $experiment_name /srv/master-thesis-alj95/code/ragnroll_project/.venv/bin/python -m ragnroll run-evaluations $config_path $evaluation_data_path $corpus_path $output_path --no-baselines --test-size=$test_size

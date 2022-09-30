conda create -n a2q2 python=3.7
source ~/anaconda3/etc/profile.d/conda.sh
conda activate a2q2
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c pytorch torchtext==0.9.0
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
conda install torchvision -c pytorch
conda install pillow=6.2.1
conda install numpy 
conda install scipy
conda install tqdm
conda install docopt
conda install nltk
pip install sentencepiece
pip install sacrebleu
pip install tensorboard
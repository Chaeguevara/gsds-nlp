conda config --add channels conda-forge
conda install zip
rm -f 2000_00000_coding.zip #change here to your student id
zip -r 2000_00000_coding.zip ./q2/*.py ./q2/model.bin ./q2/model.bin.optim ./q3/*.py ./q3/*.pt
#change above to your student id
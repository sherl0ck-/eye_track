CUR=$(pwd)
cd "/Users/Brinck/Documents/New York University Abu Dhabi/Courses/Cap Stone/Eye Tracking/preprocessing/build"
make
./bin/eyeLike

cd "/Users/Brinck/Documents/data/preprocessed"
cp data $CUR"/data-final50000.txt"

cd $CUR
echo
python svrFred.py data-final50000.txt 1 0
python svrFred.py data-final50000.txt 0 0

python svrFred.py data-final50000.txt 1 1
python svrFred.py data-final50000.txt 0 1
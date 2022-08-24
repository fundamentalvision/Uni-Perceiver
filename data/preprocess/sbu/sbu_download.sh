#!/bin/bash

export http_proxy=http://172.16.1.135:3128/ ; export https_proxy=http://172.16.1.135:3128/ ; export HTTP_PROXY=http://172.16.1.135:3128/ ; export HTTPS_PROXY=http://172.16.1.135:3128/


srun -p cpu --cpus-per-task 20 \
cat dataset/download_list.txt | xargs -n 2 -P 20 wget -nc -U 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17' --timeout=1 --waitretry=0 --tries=5 --retry-connrefused -nv -O


find ./images -type f -size -1c -exec rm {} \;
ls -d ./images/* | xargs -n 1 -P 20 python check_valid.py | tee image_size_invalid.txt
xargs rm < image_size_invalid.txt
rm image_size_invalid.txt
ls ../image > image_valid.txt
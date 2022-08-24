# Prepare Data

* By default,   all training data used for this repository will be searched  in the directory `DATA_PATH`. Please specify or change your data location before code running as following:

    ```
    export DATA_PATH='/mnt/lustre/share_data/zhujinguo/open_source_dataset'
    ```


* To make it very easy for you to run the training code, we provide [a toy dataset](https://drive.google.com/file/d/14GZPYqVLiXVYjxRGvC9WO4WK3WiwBPRY/view?usp=sharing), which is a small subset of our pretraing data.
You should download this and unzip this file to `DATA_PATH`.

    With the data of this subset, you can train Uni-Perceiver with  config file [configs/BERT_L12_H192_experiments/4tasks_training_small_datasets.yaml](../configs/BERT_L12_H192_experiments/4tasks_training_small_datasets.yaml).
    Please refer to [pretraining.md](./pretraining.md) for training usage.

* For tasks with a fixed candidate target sets, such as image / video classification (where the target sets are the category labels) and masked language modeling (where the target set is the vocabulary), you also need to perpare the target set file. Please refer to the jupyter notebook [tools/generate_target_sets.ipynb](../tools/generate_target_sets.ipynb) for details.

* For the complete datasets for training our models, please download datasets according to the instructions below:

## Different datasets

### Todo List:
- [x] Imagenet-21k and Imagenet-1k
- [x] books&wiki
- [x] MSCOCO Caption
- [x] YFCC 
- [x] CC12M 
- [x] CC3M 
- [x] Visual Genome
- [x] SBU 
- [x] Kinetics-400 & Kinetics-700
- [x] Moments in Time 
- [x] Flickr30k
- [x] MSVD 
- [x] MSR-VTT
- [x] GLUE 
- [x] VQA 

### Imagenet-1k

1. Please download the images of imagenet dataset from the official website [Imagenet](https://image-net.org/).

2. We provide the annotation files (including train.txt, val.txt and test.txt) on [meta](https://drive.google.com/file/d/1piqII0qGHmK1pop0RjdoFx927hcm1Mny/view).

3. a) Tokenizing imagenet class names to generate "imagenet_class_name_CLIP_with_endoftext.pkl" using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb)

   b) Or using generated file we provide from [here](https://drive.google.com/file/d/1bgFohNsppe7kksxTbWgSFMoEZuX0szc_/view?usp=sharing) 
4. Organize them as follows:
    ```
    DATA_PATH/
    └── imagenet/
        ├── imagenet_class_name_CLIP_with_endoftext.pkl
        ├── meta
        │   ├── test.txt
        │   ├── train.txt
        │   └── val.txt
        ├── test
        │   ├── ILSVRC2012_test_00000001.JPEG
        │   ├── ILSVRC2012_test_00000002.JPEG
        │   ├── ILSVRC2012_test_00000003.JPEG
        │   ├── ILSVRC2012_test_00000004.JPEG
        │   └── ...
        ├── train
        │   ├── n01440764
        │   │   ├── n01440764_10026.JPEG
        |   │   ├── n01440764_10027.JPEG
        |   │   ├── n01440764_10029.JPEG
        |   │   └── ...
        │   ├── n01443537
        |   │   └── ...   
        │   ├── n01484850
        |   │   └── ...
        |   └── ...
        └─── val
           ├── ILSVRC2012_val_00000001.JPEG
           ├── ILSVRC2012_val_00000002.JPEG
           ├── ILSVRC2012_val_00000003.JPEG
           └── ...

    ```






### Imagenet-22k
1. Please refer to Imagenet-1K dataset. 

2. Meta file is provided from [here](https://drive.google.com/file/d/1TDF0i8tXTB-K-zYOVhsmmocAtgrKpOG8/view?usp=sharing)

3. Imagenet class name file in [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb) for tokenizing is provided from [here](https://drive.google.com/file/d/1cJHD5Ysxfr4tRMqAAwjOah2glfFktiT1/view?usp=sharing). Or you can directly use the CLIP-tokenized imagenet-22K class name files is provided from [here](https://drive.google.com/file/d/1juSGVP8IjERXoM-AwxKRDtLk65p9FTds/view?usp=sharing)

### Books&wiki
1. please download files [wiki.doc](https://drive.google.com/file/d/1rZJ-Nj_SSqwu85tME3wbN8tfGhljfAsf/view) abd [bc1g.doc](https://drive.google.com/file/d/16T5EYqIjO-tAj1OFxz6bnnzEABCusCcv/view).
And  put them together into a file:
    ```
    cat wiki.doc bc1g.doc > bookswiki.doc
    ```
2. <a id="vocab"></a>  a) Tokenizing vocabularies to generate "vocabulary_CLIP_with_endoftext.pkl" using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb)

   b) Or using generated file we provide from [here](https://drive.google.com/file/d/1omEahjKjeWe0a4PSXEHaGE_WVdiZLf4W/view?usp=sharing) 

3. Then put this files in `DATA_PATH`
    ```
    DATA_PATH/
    ├── vocabulary_CLIP_with_endoftext.pkl
    └── bert_pretrain_data/
        └─ bookswiki/
        └── bookswiki.doc
            
    ```
4. you can also download the plain text dataset from [huggingface.co/datasets/wikipedia](https://huggingface.co/datasets/wikipedia) and [huggingface.co/datasets/bookcorpus](https://huggingface.co/datasets/bookcorpus).

### MSCOCO

1. Please download the images of COCO 2014 from [MSCOCO](https://cocodataset.org/#download).
2. Download preprocessed coco captions from Karpathy's homepage: [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) and extract "dataset_coco.json" from zip file.

3. a) You can run the [coco_preprocess.py](./preprocess/coco_preprocess.py) file to split the dataset_coco.json file into train, val and test part:
    
    1. walk into the /data/preprocess folder and open the [coco_preprocess.py](./preprocess/coco_preprocess.py) file;
    2. fill the 'original_json' variable with the path you download the dataset_coco.json file.
    3. fill the 'savepath' with the path you want to save the splited json file.
    4. run the [coco_preprocess.py](./preprocess/coco_preprocess.py) file.
    
    b) Or you can directly use the generated json file we provide from [here](https://drive.google.com/file/d/12XUh4-Lb82RXg7Sa-Vgtut2dhIqrp7Sy/view) 

4. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
5. Organize the files into following structure:
    ```
    DATA_PATH/
    ├── vocabulary_CLIP_with_endoftext.pkl
    └── mscoco_dataset/
        ├── new_annotations
        │   ├── captions_test5k.json
        │   ├── captions_train113k.json
        │   ├── captions_val5k.json
        |   └── dataset_coco.json
        └── coco_origin 
            ├── train2014
            │   ├── COCO_train2014_000000000009.jpg
            |   ├── COCO_train2014_000000000025.jpg
            |   ├── COCO_train2014_000000000030.jpg
            │   └── ...
            └── val2014
                ├── COCO_val2014_000000000042.jpg
                ├── COCO_val2014_000000000073.jpg
                ├── COCO_val2014_000000000074.jpg
                └── ...


    ```

### Visual Genome
1. Please download the images and region decriptions of visual genome from [VG](https://visualgenome.org/api/v0/api_home.html).

2. a) You can run the [region_descriptions.ipynb](./preprocess/region_descriptions.ipynb) to preprocess the downloaded "region_descriptions.json" file:
    
    1. walk into the /data/preprocess folder and open the [region_descriptions.ipynb](./preprocess/region_descriptions.ipynb);
    2. fill the path of downloaded 'region_descriptions.json' and the path you want to save the processed file.
    3. run the [region_descriptions.ipynb](./preprocess/region_descriptions.ipynb).
    
    b) Or you can directly use the generated json file we provide from [here](https://drive.google.com/file/d/1pnl30qAPr03RpKbdbH13YZI9GtWseEHf/view) 
3. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
4. Organize the files into following structure:

    ```
        DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── visual_genome/
            ├── annotations
            │   ├── region_descriptions.json
            │   ├── vg_captions_128filter.json
            └── images 
                ├── VG_100K
                │   ├── 2.jpg
                |   ├── 3.jpg
                |   ├── 4.jpg
                │   └── ...
                └── VG_100K_2
                    ├── 1.jpg
                    ├── 51.jpg
                    ├── 52.jpg
                    └── ...


    ```

### Flickr30k
1. Please download the images of filckr30k according to the instruction of [Flickr30k](http://shannon.cs.illinois.edu/DenotationGraph/).
2. Download [flickr_jsons](https://drive.google.com/file/d/1_dJsD8_YXWtR0124X_RiEgcx1c6B_BUM/view) which provides the annotations of flickr30k images.
3. a) You can run the [process_flickr_caption_json.py](./preprocess/process_flickr_caption_json.py) to preprocess the json file:
    
    1. walk into the /data/preprocess folder and open the [process_flickr_caption_json.py](./preprocess/process_flickr_caption_json.py);
    2. fill the path of downloaded json files and fill the path you want to save the processed json files.
    3. run the [process_flickr_caption_json.py](./preprocess/process_flickr_caption_json.py).
    
    b) Or you can directly use the generated json files (including captions_test.json, captions_train.json and captions_val.json) we provide from [here](https://drive.google.com/file/d/1WIWUKbXfBJd1S0izTe_OuP7bjCFDk2wk/view) 

4. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
5. Organize the files into following structure:

    ```
        DATA_PATH/
         ├── vocabulary_CLIP_with_endoftext.pkl
         ├── flickr30k/
         │   ├── captions_test.json
         │   ├── captions_train.json
         │   └── captions_val.json
         └── flickr30k_images
                └──  flickr30k_images
                    └──  flickr30k_images
                        ├── 36979.jpg
                        ├── 65567.jpg
                        └── ...

        ```

### SBU
1. Please download the SBU [url](https://drive.google.com/file/d/1Hfbw8DVSnE3ZAaWZ7C6d6hlUde7Pr_YN/view) and [caption](https://drive.google.com/file/d/1GY_kFyiFqOHAYvjfRdM98LMlAsnFmxic/view?usp=sharing) files.
2. Filling the path of above files in [sbu_download_list.py](./preprocess/sbu/sbu_download_list.py) and run it for generating the download_list.
3. Running the script [sbu_download.sh](./preprocess/sbu/sbu_download.sh) to download the sbu images.
4. a) You can run the [make_sbu_json.py](./preprocess/sbu/make_sbu_json.py) to get the annotation file:
    
    b) Or you can directly download the generated json file [sbucaption.json](https://drive.google.com/file/d/1xFJPvyJNlH0jzqzHRN16Hk5DmKGiGEJE/view) we provide.
5. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
6. Organize the files into following structure:

    ```
        DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── sbucaption/
            ├── annotations
            │  └── sbucaptions.json
            └── images
                ├── 4385058960_b0f291553e.jpg
                ├── 5148648301_1174ef59bc.jpg
                └── ...

    ```
### CC3M
1. Please download "Train_GCC-training.tsv" and "Validation_GCC-1.1.0-Validation.tsv" from [here](https://ai.google.com/research/ConceptualCaptions/download)
2. Filling the path of "Train_GCC-training.tsv" in [cc3m_train_download_list.py](./preprocess/cc3m/cc3m_train_download_list.py) and run it for generating the training download list.
3.  Filling the path of "Validation_GCC-1.1.0-Validation.tsv" in [cc3m_val_download_list.py](./preprocess/cc3m/cc3m_val_download_list.py) and run it for generating the validation download list.
4. Running the script [cc3m_train_download.sh](./preprocess/cc3m/cc3m_train_download.sh) and [cc3m_val_download.sh](./preprocess/cc3m/cc3m_val_download.sh) to download the cc3m images.
5. Zip (without compression) "train_image", "val_image" by:
    ```
    zip -0 ../train_image.zip ./*
    zip -0 ../val_image.zip ./*

    ```
6. a) You can run the [make_cc3m_train_json.py](./preprocess/cc3m/make_cc3m_train_json.py) and [make_cc3m_val_json.py](./preprocess/cc3m/make_cc3m_val_json.py) to get the annotation file:
    
    b) Or you can directly download the generated json files [train_spacy.json](https://drive.google.com/file/d/1_bqx0xQOQC3bd40GLMC27TyLRi1tHRlC/view) and [val_spacy.json](https://drive.google.com/file/d/11ibsX_K-hgdHiomk9c6JvuAl2kYW8tjt/view) we provide.
7. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
8. Organize the files into following structure:

    ```
        DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── cc3m/
            ├── train_spacy.json
            ├── val_spacy.json
            ├──train_image
            │   ├── 00000000.jpg
            │   └── ...
            └── val_image
                ├── 00000000.jpg
                └── ...

    ```

### CC12M
1. Please download "cc12m.tsv" from [here](https://github.com/google-research-datasets/conceptual-12m)
2. Filling the path of "cc12m.tsv" in [cc12m_train_download_list.py](./preprocess/cc12m/cc12m_train_download_list.py) and run it for generating the training download list.
3. Running the script [cc12m_train_download.sh](./preprocess/cc12m/cc12m_train_download.sh) to download the cc12m images.
5. Zip (without compression) "train_image" by:
    ```
    zip -0 ../train_image.zip ./*
    ```
5. a) You can run the [make_cc12m_train_json.py](./preprocess/cc12m/make_cc12m_train_json.py) to get the annotation file:
    
    b) Or you can directly download the generated json file [train_available.json](https://drive.google.com/file/d/1SVHmHpewvmpCbWDCsLSbwQ8lhusQXEIt/view) we provide.
6. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
7. Organize the files into following structure:

    ```
        DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── c12m/
            ├── train_available.json
            └── train_image
                ├── 00000000.jpg
                └── ...

    ```

### Kinetics-400 & Kinetics-700
1. Please download the Kinectics-400 & Kinetics-700 videos according to the instructions of [this](https://github.com/cvdfoundation/kinetics-dataset)

2. a) 

    i. Filling the path of K400's "training" and "validation" folder you download in [k400_construct_csv.py](./preprocess/k400_construct_csv.py) and run it for generating the K400 related files (K400_val.csv, K400_train.csv, categories.txt, annotation.json).

    ii. Filling the path of K700's "training" and "validation" folder you download in [k700_construct_csv.py](./preprocess/k700_construct_csv.py) and run it for generating the K700 related files (K700_val.csv, K700_train.csv, categories.txt, annotation.json).

    iii. Running script [video_categories.ipynb](../tools/video_categories.ipynb) to generate "category_mapping.txt".

    b) Or you can directly download the processed files we provide: [K400](https://drive.google.com/file/d/1YqchifEjoovZYJ77Egn5pHv3E1olRIpq/view?usp=sharing), [K700](https://drive.google.com/file/d/1fHdcBRdU27w7OfNijP0ZBsNbxLQSLfRa/view?usp=sharing)

3. a) Tokenizing K400, K700 class names to generate "k400_class_name_CLIP_with_endoftext.pkl" and "k700_class_name_CLIP_with_endoftext.pkl" using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb)

   b) Or using generated file we provide from [K400-CLIP](https://drive.google.com/file/d/1V-SpRzugmFgHR6j7ifFLqh7Ao5VW-gM8/view?usp=sharing) and [K700-CLIP](https://drive.google.com/file/d/1lq9WaEWh1lmfBv4pOs8Aj9yTrTxkQc3Z/view?usp=sharing)
   
   
4. Organize the files into following structure:
    ```
    DATA_PATH/
        ├── k400_class_name_CLIP_with_endoftext.pkl
        └── K400/
            ├── training
            │    ├── abseiling
            │    │   ├── _4YTwq0-73Y_000044_000054.mp4
            │    │   └── ...
            │    ├── air_drumming
            │    └── ...
            ├── validation/
            │    ├── abseiling
            │    │   ├── __NrybzYzUg.mkv
            │    │   └── ...
            │    ├── air_drumming
            │    └── ...
            ├── annotation.json
            ├── category_mapping.txt
            ├── categories.txt
            ├── K400_train.csv
            └── K400_val.csv
    ```
    K700 is similar.

### MomentsInTime

1. Please download the MomentsInTime videos according to the instructions of [Official Website](http://moments.csail.mit.edu/)

2. a) 

    i. Filling the path of "training" folder you download in [moments_construct_csv.py](./preprocess/moments_construct_csv.py) and run it for generating the training files (moments_train.csv, categories.txt, annotation.json).

    ii. Running script [video_categories.ipynb](../tools/video_categories.ipynb) to generate "category_mapping.txt".

    b) Or you can directly download the processed files we provide: [moments](https://drive.google.com/file/d/1aXVCBKrocatZfT8TRKv4TuxHkTa7SxMz/view?usp=sharing).
3. a) Tokenizing momentsInTime class names to generate "MiT_class_name_CLIP_with_endoftext.pkl" using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb)

   b) Or using generated file we provide from [MiT-CLIP](https://drive.google.com/file/d/1xNC8Dld-0x735nO60cwUBYG3gTVvoPC8/view?usp=sharing)

4. Organize the files into following structure:
    ```
    DATA_PATH/
        ├── MiT_class_name_CLIP_with_endoftext.pkl
        └── MomentsInTime/
            ├── training
            │    ├── adult+female+singing
            │    │   ├── 0a2b81cb0ec5fde79b8c.mp4
            │    │   └── ...
            │    ├── adult+female+speaking
            │    └── ...
            ├── annotation.json
            ├── categories.txt
            ├── category_mapping.txt
            └── moments_train.csv
    ```


### MSVD
1. Download MSVD videos "YoutTubeClips.tar" from [here](https://www.cs.utexas.edu/users/ml/clamp/videoDescription/) and preprocessed "txt_labels" from [here](https://github.com/nasib-ullah/video-captioning-models-in-Pytorch/tree/main/MSVD/captions).
2. a) Fill the path of downloaded files in [msvd_preprocess.py](./preprocess/msvd_preprocess.py)  to generate the annotation files (caption_msvd_train_cocostyle.json, caption_msvd_val_cocostyle.json, caption_msvd_test_cocostyle.json)
    
    b) Or directly download the annotation files we provide [new_annotations](https://drive.google.com/file/d/1VHT8waNVp8LUFlfY_YACCbVrzBQW3sWy/view)

3. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
4. Organize the files into following structure:
    ```
    DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── msvd_dataset/
            ├── new_annotations
            │    ├── caption_msvd_test_cocostyle.json
            │    ├── caption_msvd_train_cocostyle
            │    └── caption_msvd_val_cocostyle
            ├── txt_labels
            │    ├── sents_test_lc_nopunc.txt
            │    ├── sents_train_lc_nopunc.txt
            │    ├── sents_train_lc_nopunc.txt
            │    └── youtube_mapping.txt
            └── YouTubeClips
                 ├── _0nX-El-ySo_83_93.avi
                 └── ...
    ```
### MSR-VTT
1. Download MSRVTT videos ("train_val_videos.zip", "test_videos.zip") and  annotation files ("train_val_annotation.zip", "test_videodatainfo.zip") from [here](https://www.mediafire.com/folder/h14iarbs62e7p/shared) and download dataset split info from [here](https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip).
2. Unzip downloaded files above, fill the paths of "test_videodatainfo.json", "train_val_videodatainfo.json", "MSRVTT_train.9k.csv", "MSRVTT_JSFUSION_test.csv" in the  [msrvtt_dataprocess_1k.ipynb](./preprocess/msrvtt_dataprocess_1k.ipynb) 
    
    b) Or directly download the annotation files ("caption_msrvtt_1k_trainval_cocostyle.json","caption_msrvtt_1k_test_cocostyle.json") we provide [annotations_new](https://drive.google.com/file/d/1ZnA4hEic6x9D7dfaEUPoa6MlQ30rITom/view)
3. Generating tokenized vocabularies as mentioned in [bookswiki part](#vocab)
4. Organize the files into following structure:
    ```
    DATA_PATH/
        ├── vocabulary_CLIP_with_endoftext.pkl
        └── msrvtt_dataset/
            ├── annotations_new
            │    ├── caption_msrvtt_1k_trainval_cocostyle.json
            │    └── caption_msrvtt_1k_test_cocostyle.json
            └── videos
                 ├── video0.mp4
                 └── ...
    ```

### VQA

1. Download VQA meta data from the datalink [vilbert](https://github.com/jiasenlu/vilbert_beta/tree/master/data) provided, files including:
    - dictionary.pkl
    - train_ids.pkl
    - val_ids.pkl
    - train_target.pkl
    - trainval_ans2label.pkl
    - val_target.pkl
    - trainval_label2ans.pkl

2. Download VG questions and answers from [here](https://drive.google.com/drive/folders/10XHRXg07lNbdZQrREhOLYVM3N0LrkTxB)
    
    
    
    
    

3. Download VQA annotations from the [link](https://visualqa.org/download.html) xmodaler provided, files including:
    - vg_target.pkl
    - VG_questions2.json
    - download
    - VG_annotations.json
4. Download VQA annotations from [VQA](https://visualqa.org/download.html) website, files including:   
    - v2_OpenEnded_mscoco_test2015_questions.json
    - v2_OpenEnded_mscoco_train2014_questions.json
    - v2_OpenEnded_mscoco_val2014_questions.json

5. a) Tokenizing all the possible answers using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb).

    b) Or you can use the tokenized answers we provide [VQA_Answers](https://drive.google.com/file/d/1X-1blHh2MrYhDq9bkdndNVRZ-49VCsuz/view?usp=sharing).
6. Organize the files into following structure:
    ```
    DATA_PATH/
    ├── vocabulary_CLIP_with_endoftext.pkl
    ├── mscoco_dataset/
    |    └── coco_origin 
    |       ├── train2014
    |       │   ├── COCO_train2014_000000000009.jpg
    |       |   ├── COCO_train2014_000000000025.jpg
    |       |   ├── COCO_train2014_000000000030.jpg
    |       │   └── ...
    |       └── val2014
    |           ├── COCO_val2014_000000000042.jpg
    |           ├── COCO_val2014_000000000073.jpg
    |           ├── COCO_val2014_000000000074.jpg
    |           └── ...
    └── VQA
        ├── trainval_ans2label.pkl
        ├── trainval_label2ans.pkl
        ├── v2_OpenEnded_mscoco_train2014_questions.json
        ├── v2_OpenEnded_mscoco_val2014_questions.json
        ├── v2_OpenEnded_mscoco_test-dev2015_questions.json
        ├── val_target.pkl
        ├── VG_questions2.json
        ├── vg_target.pkl
        └── coco_map.json


    ```
### GLUE
1. Follow the instructions of [this](https://github.com/nyu-mll/GLUE-baselines) to download and preprocess GLUE benchmark data.
2. a) Tokenizing GLUE datasets using [generate_target_sets.ipynb](../tools/generate_target_sets.ipynb).

    b) Or you can use the tokenized answers we provide [GLUE_classnames](https://drive.google.com/file/d/1HR7xHsIRsS4iUwGr3CX6h5z_dVn-EJt-/view?usp=sharing).

3. Organize the files into following structure:
```    
    DATA_PATH/
        ├── GLUE_classnames
        └── bert_pretrain_data/
            └── glue_data
                 ├── CoLA
                 ├── CoLA-bin
                 ├── diagnostic
                 ├── MNLI
                 ├── MNLI-bin
                 ├── MRPC
                 ├── MRPC-bin
                 ├── QNLI
                 ├── QNLI-bin
                 ├── QQP
                 ├── QQP-bin
                 ├── RTE
                 ├── RTE-bin
                 ├── SST-2
                 ├── SST-2-bin
                 ├── STS-B
                 ├── STS-B-bin
                 └── WNLI
```
    
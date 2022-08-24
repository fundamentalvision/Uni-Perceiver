f = open('dataset/SBU_captioned_photo_dataset_urls.txt', 'r')

download_list = 'dataset/download_list.txt'
out = open(download_list, 'w')

image_path = '/mnt/lustrenew/share_data/zhujinguo/SBU/images'
for url in f.readlines():
    url = url.strip()
    filename = url.split('/')[-1]
    out.write("{path}/{filename}\t{url}\n".format(path=image_path, filename=filename, url=url))
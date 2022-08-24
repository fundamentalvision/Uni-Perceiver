## data structure

* imagenet 1k 

```  
data = {
    'input_sample_list': [
        {
            'data':
            torch.rand(bs, 3, 224, 224, dtype=torch.float32),
            'invalid_mask':
            None,
            'modality':
            'image',
            'data_type': 'input',
            'sample_info': {
                'id': list(range(bs)),
                'path': ['hah' for _ in range(bs)]
            }
        },
    ],
    'target_sample_list': [],
    'target_idx_list': [torch.randint(0, 1000, (bs, ))],
    'target_set_list': ['ImageNet22k'],
    'shared_target_sets': {
        'ImageNet22k': [{
            'data':
            torch.randint(0, 49411, (1000, 11)),
            'invalid_mask':
            torch.zeros(1000, 11, dtype=torch.bool),
            'modality':
            'text',
            'data_type': 'target',
            'sample_info': {
                'distributed': True,
                'total_num': 1000,
            }
        }]
    },
    'task_info': {
        'task_name': 'imagenet',
        'task_type': 'image_classification',
        'dataset_name': 'ImageNet22k',
        'batchsize': None,
        'sampling_ratio': None
    }
}
```
* mscoco caption
```           data = {
    'input_sample_list': [
        {
            'data':
            torch.rand(bs, 3, 224, 224, dtype=torch.float32),
            'invalid_mask':
            None,
            'modality':
            'image',
            'data_type': 'input',
            'sample_info': [{
                'id': id,
                'path': 'hahah',
                'bs': bs
            } for _ in range(bs)]
        },
        {
            'data':
            torch.randint(0, 49411, (bs, 31 * 2)),
            'invalid_mask':
            torch.zeros(bs, 31 * 2, dtype=torch.bool),
            'modality':
            'text',
            'data_type': 'input',
            'sample_info': [{
                'pe_index':
                torch.cat([torch.arange(31),
                            torch.arange(31)],
                            dim=0)
            } for _ in range(bs)]
        },
    ],
    'target_sample_list': [],
    'target_idx_list': [torch.randint(0, 49411, (bs, 31))],
    'target_set_list': ['Vocab_Word'],
    'shared_target_sets': {
        'Vocab_Word': [{
            'data': torch.randint(0, 49411, (49411, 2)),
            'invalid_mask': None,
            'modality': 'text',
            'data_type': 'target',
            'sample_info': {
                'distributed': True,
                'total_num': 49411,
            }
        }]
    },
    'task_info': {
        'task_name': 'mscoco_caption',
        'task_type': 'image_caption',
        'dataset_name': 'MSCOCO',
        'batchsize': None,
        'sampling_ratio': None
    }
}
```


*  text_mlm
```
data = {
    'input_sample_list': [
        {
            'data': torch.randint(0, 49411, (bs, 128)),
            'invalid_mask': torch.zeros(bs, 128, dtype=torch.bool),
            'modality': 'text',
            'data_type': 'input',
            'sample_info': {
                'seq_length': 128
            }
        },
    ],
    'target_sample_list': [],
    'target_idx_list': [torch.randint(0, 49411,
                                        (bs, 128))],  # most are -1,
    'target_set_list': ['Vocab_Word'],
    'shared_target_sets': {
        'Vocab_Word': [{
            'data': torch.randint(0, 49411, (49411, 2)),
            'invalid_mask': None,
            'modality': 'text',
            'data_type': 'target',
            'sample_info': {
                'distributed': True,
                'total_num': 49411,
            }
        }]
    },
    'task_info': {
        'task_name':  'bookswiki_pretrain',
        'task_type': 'text_mlm',
        'dataset_name': 'BooksWiki',
        'batchsize': None,
        'sampling_ratio': None
    }
}
```


 * mscoco retrieval
 ```
data = {
    'input_sample_list': [
        {
            'data':
            torch.rand(bs, 3, 224, 224, dtype=torch.float32),
            'invalid_mask':
            None,
            'modality':
            'image',
            'sample_info': {
                'id': list(range(bs)),
                'path': ['hah' for _ in range(bs)]
            }
        },
    ],
    'target_sample_list': [
        {
            'data': torch.randint(0, 49411, (bs, 30)),
            'invalid_mask': torch.zeros(bs, 30,
                                        dtype=torch.bool),
            'modality': 'text',
            'sample_info': {}
        },
    ],
    'target_idx_list': [],
    'target_set_list': [],
    'shared_target_sets': {
        'ImageNet22k': [{
            'data':
            torch.randint(0, 49411, (1000, 11)),
            'invalid_mask':
            torch.zeros(1000, 11, dtype=torch.bool),
            'modality':
            'text',
            'sample_info': {
                'distributed': True,
                'total_num': 1000,
            }
        }]
    },
    'task_info': {
        'task_name': 'mscoco_retrieve',
        'task_type': 'image_retrieval',
        'dataset_name': 'MSCOCO',
        'batchsize': None,
        'sampling_ratio': None
    }
}
```
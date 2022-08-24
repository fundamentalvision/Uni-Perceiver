import torch

def get_moe_embedding(moe_type):

    if moe_type == 'attribute':
        Task_attribute = {
            # task input -- TASK_TYPE & data_type
            'image_classification': {
                "input":
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
            },
            'video_classification': {
                "input":
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
            },
            'text_mlm': {
                "input":
                torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float),
            },
            'image_caption': {
                "input":
                torch.tensor(
                    [[1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1]],
                    dtype=torch.float)
            },
            'video_caption': {
                "input":
                torch.tensor(
                    [[1, 1, 0, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 1, 0, 1]],
                    dtype=torch.float)
            },
            'image_retrieval': {
                'input':
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
                'target':
                torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            },
            'video_retrieval': {
                'input':
                torch.tensor([[1, 0, 0, 1, 1, 0, 0, 0]], dtype=torch.float),
                'target':
                torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            },
            'text_classification': {
                "input":
                torch.tensor([[0, 1, 0, 1, 0, 1, 0, 0]], dtype=torch.float),
            },
            

            # SHARED_TARGETS
            "ImageNet1k":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "ImageNet22k":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MomentsInTime":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Kinetics700":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Kinetics400":
            torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "Vocab_Word":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "CoLA-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MNLI-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "MRPC-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "QNLI-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "QQP-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "RTE-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
            "SST-2-target":
            torch.tensor([[1, 1, 0, 1, 0, 1, 1, 0]], dtype=torch.float),
        }
        return Task_attribute
    else:
        raise NotImplementedError(f'please check MOE_TYPE {moe_type}')



def get_embed_with_task_type(moe_type: str, task_type: str, data_type: str):
    if moe_type is None:
        return None
    embedding_dict = get_moe_embedding(moe_type)
    return embedding_dict[task_type][data_type]


def get_embed_with_shared_tagert_name(moe_type: str, set_name: str,):
    if moe_type is None:
        return None
    embedding_dict = get_moe_embedding(moe_type)
    return embedding_dict[set_name]
